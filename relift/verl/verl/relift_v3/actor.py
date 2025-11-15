# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReLIFTDataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, sft_actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.sft_actor_optimizer = sft_actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

        self.before_sft_grad_norm = None

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite or larger than max_grad_skip, skip the update
        if not torch.isfinite(grad_norm) or (grad_norm > self.config.max_grad_norm):
            print("RL grad set to zero")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm
    
    def _sft_optimizer_step(self):
        assert self.config.sft.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.sft.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.sft.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.sft.grad_clip)

        # if grad_norm is not finite, skip the update
        if (not torch.isfinite(grad_norm)) or (grad_norm > self.config.sft.max_sft_grad_norm) \
                or (self.config.sft.using_dynamic_max_sft_grad_norm and (self.before_sft_grad_norm is not None) and grad_norm > 1.5 * self.before_sft_grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite or too large: {grad_norm}")
            self.sft_actor_optimizer.zero_grad()
        else:
            self.before_sft_grad_norm = grad_norm
            self.sft_actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_relift(self, data: DataProto, sft_data: DataProto):
        # 导入 SFT loss (原 SFT 函数中在循环内导入)
        from .core_algos import compute_sft_loss

        # 确保处于训练模式
        self.actor_module.train()

        # --- 1. 设置 (主要使用 RL data 的 meta_info) ---
        temperature = data.meta_info["temperature"]  # temperature 必须在 data.meta_info 中
        multi_turn = data.meta_info.get("multi_turn", False)
        
        # SFT 也需要 multi_turn, 确认 sft_data 中是否也需要
        # 为简化起见，我们假设 RL data 的 multi_turn 适用于两者
        sft_multi_turn = sft_data.meta_info.get("multi_turn", False)
        # 您可能需要根据实际情况调整 multi_turn 的逻辑，这里我们分别使用
        
        metrics = {}

        # --- 2. RL (PPO) Dataloader 设置 ---
        select_keys_rl = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys_rl.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys_rl.append("ref_log_prob")
        
        has_multi_modal_inputs_rl = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # --- 3. SFT Dataloader 设置 ---
        select_keys_sft = ["responses", "input_ids", "attention_mask", "position_ids"]
        if sft_multi_turn: # 使用 SFT 自己的 multi_turn 标志
             select_keys_sft.append("loss_mask")
             
        has_multi_modal_inputs_sft = "multi_modal_inputs" in sft_data.non_tensor_batch.keys()

        # --- 4. 创建并行的 Dataloaders (假设 PPO 的配置主导) ---
        # 关键假设：data 和 sft_data 具有相同的批次大小
        # 并且我们将使用 PPO 的 mini_batch_size 来切分两者
        
        # 检查多模态输入是否一致，这对于并行处理至关重要
        if has_multi_modal_inputs_rl != has_multi_modal_inputs_sft:
            raise ValueError("RL data and SFT data must have the same multi-modal status for relift update.")
        
        has_multi_modal_inputs = has_multi_modal_inputs_rl

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            
            dataloader_rl = data.select(select_keys_rl, non_tensor_select_keys).chunk(num_mini_batches)
            dataloader_sft = sft_data.select(select_keys_sft, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            batch_rl = data.select(batch_keys=select_keys_rl).batch
            batch_sft = sft_data.select(batch_keys=select_keys_sft).batch
            
            dataloader_rl = batch_rl.split(self.config.ppo_mini_batch_size)
            dataloader_sft = batch_sft.split(self.config.ppo_mini_batch_size) # 使用 PPO 的 mini_batch_size

        # --- 5. 训练循环 (使用 PPO epochs) ---
        for epoch in range(self.config.ppo_epochs):
            # 并行遍历 mini-batches
            for batch_idx, (rl_data_chunk, sft_data_chunk) in enumerate(zip(dataloader_rl, dataloader_sft)):
                
                # --- 6. 并行 Micro-batch 切分 (使用 PPO 配置) ---
                rl_mini_batch = rl_data_chunk
                sft_mini_batch = sft_data_chunk

                if has_multi_modal_inputs:
                    # RL PPO 的逻辑
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = rl_mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    
                    rl_micro_batches = rl_data_chunk.select(select_keys_rl, non_tensor_select_keys).chunk(num_micro_batches)
                    sft_micro_batches = sft_data_chunk.select(select_keys_sft, non_tensor_select_keys).chunk(num_micro_batches)
                
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    
                    rl_micro_batches, _ = rearrange_micro_batches(batch=rl_mini_batch, max_token_len=max_token_len)
                    sft_micro_batches, _ = rearrange_micro_batches(batch=sft_mini_batch, max_token_len=max_token_len)
                
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    
                    rl_micro_batches = rl_mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    sft_micro_batches = sft_mini_batch.split(self.config.ppo_micro_batch_size_per_gpu) # 使用 PPO 的 micro_batch_size

                # 使用 PPO (RL) 的优化器
                self.actor_optimizer.zero_grad()

                # --- 7. 并行 Micro-batch 处理 ---
                for rl_micro_data, sft_micro_data in zip(rl_micro_batches, sft_micro_batches):
                    
                    # --- 7a. 处理 RL (PPO) Loss ---
                    if isinstance(rl_micro_data, DataProto):
                        data_rl = {**rl_micro_data.batch.to(get_torch_device().current_device()), **rl_micro_data.non_tensor_batch}
                    else:
                        data_rl = rl_micro_data.to(get_torch_device().current_device())
                    
                    responses_rl = data_rl["responses"]
                    response_length_rl = responses_rl.size(1)
                    attention_mask_rl = data_rl["attention_mask"]
                    
                    if multi_turn:
                        response_mask_rl = data_rl["loss_mask"][:, -response_length_rl:]
                    else:
                        response_mask_rl = attention_mask_rl[:, -response_length_rl:]

                    old_log_prob = data_rl["old_log_probs"]
                    advantages = data_rl["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff_rl = self.config.entropy_coeff # RL 熵系数
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy_rl = (entropy_coeff_rl != 0)
                    entropy_rl, log_prob_rl = self._forward_micro_batch(micro_batch=data_rl, temperature=temperature, calculate_entropy=calculate_entropy_rl)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_rl,
                        advantages=advantages,
                        response_mask=response_mask_rl,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff_rl != 0:
                        entropy_loss_rl = agg_loss(loss_mat=entropy_rl, loss_mask=response_mask_rl, loss_agg_mode=loss_agg_mode)
                        policy_loss_rl = pg_loss - entropy_loss_rl * entropy_coeff_rl
                    else:
                        policy_loss_rl = pg_loss
                        entropy_loss_rl = 0 # 用于记录

                    if self.config.use_kl_loss:
                        ref_log_prob = data_rl["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob_rl, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask_rl, loss_agg_mode=loss_agg_mode)
                        
                        policy_loss_rl = policy_loss_rl + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        loss_rl = policy_loss_rl * (len(data_rl) / self.config.ppo_mini_batch_size)
                    else:
                        loss_rl = policy_loss_rl / self.gradient_accumulation

                    # 记录 RL metrics
                    rl_metrics_data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/rl_loss_scaled": float(loss_rl.detach().cpu().item()),
                    }
                    if entropy_coeff_rl != 0:
                        rl_metrics_data['actor/rl_entropy_loss'] = float(entropy_loss_rl.detach().cpu().item())
                    append_to_dict(metrics, rl_metrics_data)


                    # --- 7b. 处理 SFT Loss ---
                    if isinstance(sft_micro_data, DataProto):
                        data_sft = {**sft_micro_data.batch.to(get_torch_device().current_device()), **sft_micro_data.non_tensor_batch}
                    else:
                        data_sft = sft_micro_data.to(get_torch_device().current_device())

                    responses_sft = data_sft["responses"]
                    response_length_sft = responses_sft.size(1)
                    attention_mask_sft = data_sft["attention_mask"]

                    if sft_multi_turn:
                        response_mask_sft = data_sft["loss_mask"][:, -response_length_sft:]
                    else:
                        response_mask_sft = attention_mask_sft[:, -response_length_sft:]

                    entropy_coeff_sft = self.config.sft.entropy_coeff # SFT 熵系数
                    loss_type_sft = self.config.sft.get("sft_loss_type", "v0")

                    calculate_entropy_sft = (entropy_coeff_sft != 0)
                    entropy_sft, log_prob_sft = self._forward_micro_batch(micro_batch=data_sft, temperature=temperature, calculate_entropy=calculate_entropy_sft)
                    
                    if loss_type_sft == "v0":
                        loss_fn = compute_sft_loss
                        ret_dict = loss_fn(log_prob=log_prob_sft, eos_mask=response_mask_sft)
                    else:
                        raise ValueError(f"Invalid sft loss type: {loss_type_sft}")

                    sft_loss = ret_dict["sft_loss"]

                    if entropy_coeff_sft != 0:
                        loss_agg_mode_sft = self.config.loss_agg_mode # 假设 SFT 和 PPO 使用相同的 agg_mode
                        entropy_loss_sft = agg_loss(loss_mat=entropy_sft, loss_mask=response_mask_sft, loss_agg_mode=loss_agg_mode_sft)
                        policy_loss_sft = sft_loss - entropy_loss_sft * entropy_coeff_sft
                    else:
                        policy_loss_sft = sft_loss
                        entropy_loss_sft = 0

                    # 使用 PPO (RL) 的缩放逻辑
                    if self.config.use_dynamic_bsz:
                        loss_sft = policy_loss_sft * (len(data_sft) / self.config.ppo_mini_batch_size) # 使用 PPO 的 mini batch size
                    else:
                        loss_sft = policy_loss_sft / self.gradient_accumulation # 使用 PPO 的 gradient accumulation

                    # 记录 SFT metrics
                    sft_metrics_data = {
                        'actor/sft_loss': float(sft_loss.detach().cpu().item()),
                        'actor/sft_loss_scaled': float(loss_sft.detach().cpu().item())
                    }
                    if entropy_coeff_sft != 0:
                        sft_metrics_data['actor/sft_entropy_loss'] = float(entropy_loss_sft.detach().cpu().item())
                    append_to_dict(metrics, sft_metrics_data)
                    

                    # --- 7c. 合并 Loss 并反向传播 ---
                    combined_loss = loss_rl + loss_sft
                    combined_loss.backward()

                    # 记录 combined loss
                    append_to_dict(metrics, {'actor/combined_loss_scaled': float(combined_loss.detach().cpu().item())})

                # --- 8. 优化器步骤 (在 micro-batch 循环之后) ---
                # 使用 PPO (RL) 的优化器步骤
                grad_norm = self._optimizer_step() 
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        
        # --- 9. 清理 ---
        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def sft_update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.sft.sft_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.sft.sft_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.sft.sft_mini_batch_size // self.config.sft.sft_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.sft.sft_micro_batch_size_per_gpu)

                self.sft_actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    entropy_coeff = self.config.sft.entropy_coeff

                    loss_type = self.config.sft.get("sft_loss_type", "v0")
                    
                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    from .core_algos import compute_sft_loss

                    if loss_type == "v0":
                        loss_fn = compute_sft_loss
                        ret_dict = loss_fn(log_prob=log_prob, eos_mask=response_mask)
                    else:
                        raise ValueError(f"Invalid sft loss type: {loss_type}")

                    sft_loss = ret_dict["sft_loss"]

                    if entropy_coeff != 0:
                        loss_agg_mode = self.config.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = sft_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = sft_loss
                        entropy_loss = 0

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.sft.sft_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation

                    loss.backward()

                    if entropy_coeff != 0:
                        data = {
                            'actor/combined_loss': float(policy_loss.detach().cpu().item()),
                            'actor/sft_loss': float(sft_loss.detach().cpu().item()),
                            'actor/sft_entropy_loss': float(entropy_loss.detach().cpu().item())
                        }
                    else:
                        data = {
                            'actor/combined_loss': float(policy_loss.detach().cpu().item()),
                            'actor/sft_loss': float(sft_loss.detach().cpu().item())
                        }
                    
                    append_to_dict(metrics, data)

                grad_norm = self._sft_optimizer_step()
                data = {"actor/sft_grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
                
        self.sft_actor_optimizer.zero_grad()
        return metrics
