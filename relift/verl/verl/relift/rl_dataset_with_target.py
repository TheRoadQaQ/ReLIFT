# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union, Optional

import pandas as pd
import copy 

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, ProcessorMixin
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length


import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output

from verl.utils.dataset.rl_dataset import RLHFDataset

from omegaconf import DictConfig, ListConfig

class RLHFDatasetWithTarget(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 config: DictConfig,
                 processor: Optional[ProcessorMixin] = None):
        super().__init__(parquet_files, tokenizer, config)
        
        self.max_target_length = config.max_target_length
        self.target_key = config.target_key

        # add unique_id
        self.dataframe = self.dataframe.map(lambda example, idx: {"sample_id": idx}, with_indices=True)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs

        tgt = row_dict.pop(self.target_key)
        if tgt is not None:
            tgt = tgt[0]
            if isinstance(tgt, dict):
                tgt = tgt['content']

            if not tgt.startswith('<think>\n'):
                tgt = '<think>\n' + tgt

            if raw_prompt.endswith('<think>\n') and tgt.startswith('<think>\n'):
                tgt = tgt[len('<think>\n'):]

            tgt_input_ids = self.tokenizer(tgt, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]
            tgt_input_ids = tgt_input_ids.reshape(1, -1)
        else:
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0) # empty target, will be pad to max_target_length

        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else:
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)
        row_dict['tgt_input_ids'] = tgt_input_ids

        return row_dict

    def _process_target(self, tgt: str, prompt: str, add_eos=False) -> torch.Tensor:
        if prompt.endswith('<think>\n') and tgt.startswith('<think>\n'):
            tgt = tgt[len('<think>\n'):]
        tgt_input_ids = self.tokenizer(tgt, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]
        if add_eos:
            tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([self.tokenizer.eos_token_id], device=tgt_input_ids.device, dtype=tgt_input_ids.dtype).reshape(-1)])

        tgt_input_ids = tgt_input_ids.reshape(1, -1)
        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)

        return tgt_input_ids

from verl import DataProto
class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)

import torch

class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1].item() # the output index should be int

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])

def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    token_ids = prompt_token_ids[:non_pad_index[-1][0]].tolist()
    return token_ids