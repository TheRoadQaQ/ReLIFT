#export HF_ENDPOINT=https://hf-mirror.com  
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from math_verify import parse, verify

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    # print(golden_answers)
    # print(predict_answers)
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192):
    # 数据处理
    df = pd.read_parquet(input_file)
    #df = df[:10]
    
    messages = df['prompt'].tolist()
    # if debug:
        # messages = messages[:10]
    
    #assert remove_system is True
    if remove_system:
        print('remove system')
        assert messages[0][0]['role'] == 'system'
        messages = [message[1:] for message in messages]
    answers = df['reward_model'].tolist()
    answers = [answer['ground_truth'] for answer in answers]
    # if debug:
        # answers = answers[:10]
    assert len(messages) == len(answers)
    data_sources = df['data_source'].tolist()
            
    #print(messages[0])
    outputs = generate_vllm(messages, model_path, template=template, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    # rets = {}
    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg = 0
    lens = []
    correct_lens = []
    incorrect_lens = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        answer = answers[i]
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n'):
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            
        if THOUGHT_DELIMITER_START in generated_text and THOUGHT_DELIMITER_END in generated_text:
            generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
        
        try:
            labels = labeling_responses([generated_text,], answer)
        except Exception as e:
            print(f'Error: {e}')
            labels = [False,]
        
        rets[data_sources[i]].append(labels[0])
        
        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0]
        })

        lens.append(len(generated_text))

        if labels[0]:
            correct_lens.append(len(generated_text))
        else:
            incorrect_lens.append(len(generated_text))
        
        if labels[0]:
            avg += 1
            
    print('accuracy: ', avg / len(outputs))
    
    for data_source, labels in rets.items():
        # print(data_source, len(labels))
        acc = np.array(labels).mean()
        print(f'{data_source}: {acc}')

    print('avg len: ', sum(lens)/len(lens))
    print('avg correct len: ', sum(correct_lens)/len(correct_lens))
    print('avg incorrect len: ', sum(incorrect_lens)/len(incorrect_lens))
    
    try:
        with open(output_file, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')
        # print(f'Save data: {save_data}')

def generate_vllm(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192):
    #vllm模型加载
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=8192)
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True
             )  # 替换成本地路径

    gen_prompts = []
    for i in range(len(messages)):
        cur_message = messages[i]
        if template == 'own': 
            gen_prompt = tokenizer.apply_chat_template(
                cur_message,
                tokenize=False,
                add_generation_prompt=True
        )
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[0]['content'])
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[0]['content'])
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else:
            raise ValueError(f'Invalid template: {template}')
        
        gen_prompts.append(gen_prompt)
        if i == 0:
            print('Example input: ', gen_prompt)

    outputs = llm.generate(gen_prompts, sampling_params)
    return outputs

if __name__ == "__main__":
    import fire
    fire.Fire(main)