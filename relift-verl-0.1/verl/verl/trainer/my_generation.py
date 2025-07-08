import csv
import ray
import numpy as np
import hydra
import os
import json
from tabulate import tabulate

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from verl.utils.model import compute_position_id_with_mask
import pandas as pd
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

# 引入 parse, verify
from math_verify import parse, verify

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        dataset = pd.read_parquet(config.data.output_path)
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        # 读取数据集
        dataset = pd.read_parquet(config.data.path)
        
        chat_lst = dataset[config.data.prompt_key].tolist()
        chat_lst = chat_lst[:100]
        chat_lst = [chat.tolist() for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        config_batch_size = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // config_batch_size) + 1
        output_lst = []

        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            # Repeat the batch n_samples times
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)
            # add template
            inputs = tokenizer.apply_chat_template(
                repeated_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors='pt',
                return_dict=True,
                tokenize=True
            )
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]

            if real_batch_size % dp_size != 0:
                dummy_data_size = dp_size - real_batch_size % dp_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            output = wg.generate_sequences(data)
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))
            output_lst.extend(output_text_unpad)

        # Reshape output_lst from (total_samples,) to (n_data, n_samples)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()
        dataset['responses'] = output_lst

        # Write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)

    # 计算label和acc
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    acc_total = 0
    acc_by_source = {}
    lens = []
    correct_lens = []
    incorrect_lens = []
    save_data = []

    from collections import defaultdict
    rets = defaultdict(list)

    for i in range(len(dataset)):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        ground_truth = reward_data['ground_truth']
        # 只用第一个response
        generated_text = response_lst[0]
        try:
            labels = labeling_responses([generated_text], ground_truth)
        except Exception as e:
            print(f'Error: {e}')
            labels = [False]
        rets[data_source].append(labels[0])
        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': ground_truth,
            'correctness': labels[0]
        })
        lens.append(len(generated_text))
        if labels[0]:
            correct_lens.append(len(generated_text))
            acc_total += 1
        else:
            incorrect_lens.append(len(generated_text))

    print('accuracy: ', acc_total / len(dataset))
    for data_source, labels in rets.items():
        acc = np.array(labels).mean()
        print(f'{data_source}: {acc}')
    print('avg len: ', sum(lens)/len(lens))
    print('avg correct len: ', sum(correct_lens)/len(correct_lens) if correct_lens else 0)
    print('avg incorrect len: ', sum(incorrect_lens)/len(incorrect_lens) if incorrect_lens else 0)

    # 保存详细结果
    jsonl_path = os.path.dirname(config.data.output_path)
    try:
        with open(jsonl_path, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {jsonl_path}')

def select_reward_fn(data_source):
    from deepscaler.rewards.math_reward import deepscaler_reward_fn
    return deepscaler_reward_fn

if __name__ == '__main__':
    main()