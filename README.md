# ReLIFT
Official Repository of "Learning what reinforcement learning can't"

# 🔗Links

- 📜 [Paper (arXiv)]()  
- 🤗 [HuggingFace Collection]()

---

# ✨Getting Started

## Installation

You can install ReLIFT dependencies by running the following commands:
```bash
conda create -n relift python=3.10
conda activate relift
cd ReLIFT
pip install -r requirements.txt
pip install -e .
cd verl
pip install -e .
```

If you encounter issues when installing flash-attn, we recommend you to install it here 
[flash-attn](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3). For example, we use this version. 
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Repo Structure

This repository includes:

- `ReLIFT`: Codes for training ReLIFT, interleaved with fine-tuning for hardest questions. Our main code changes are in ReLIFT/verl/verl/relift.
- `dataset`: Dataset for training and evaluating LUFFY. 
- `exp_scripts`: Example script to train LUFFY.
- `eval_scripts`: Evaluation scripts.

---

# 📖Introduction

LUFFY is a reinforcement learning framework that bridges the gap between zero-RL and imitation learning by incorporating off-policy reasoning traces into the training process. Built upon GRPO, LUFFY combines on-policy rollouts with off-policy demonstrations during advantage estimation and introduces **policy shaping** via regularized importance sampling to emphasize low-probability yet crucial actions.

![overview](./figures/luffy_performance.jpg)

### Key Highlights:
- **Off-Policy Guidance:** Seamlessly integrates external reasoning traces to bootstrap learning from stronger models.
- **Dynamic Balance:** Learns when to imitate and when to explore, adapting over the course of training.
- **Policy Shaping:** Emphasizes important actions often ignored in standard policy gradients, enabling better generalization.

---



# 🔧Usage

## Data Preparation
You need to first run the data preparation script to get the training data in parquet format.
```bash
cd data
python prepare_train.py
```

## Training

We provide an example script to train LUFFY on our subset of OpenR1-Math-220k. You can run the following command to train LUFFY:

```bash
  cd exp_scripts
  bash train.sh
```

## Inference

Here’s an example of using LUFFY for inference:

<details>
<summary>Click to view inference example</summary>

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path="Elliott/LUFFY-Qwen-Math-7B-Zero"

question = "which number is larger? 9.11 or 9.9?"

tokenizer = AutoTokenizer.from_pretrained(model_path)
messages = [{"role": "user", "content": question}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_path)
params = SamplingParams(temperature=0.6, max_tokens=8192)
outputs = llm.generate([chat], params)
print(outputs[0].outputs[0].text)
```

</details>


## Models

| **Model**                          | **Huggingface** |  **Base Model** |
|-----------------------------------|------------------|------------------|
| LUFFY-Qwen-Math-7B-Zero | https://huggingface.co/Elliott/LUFFY-Qwen-Math-7B-Zero |  Qwen2.5-Math-7B |
| LUFFY-Qwen-Math-1.5B-Zero | https://huggingface.co/Elliott/LUFFY-Qwen-Math-1.5B-Zero | Qwen2.5-Math-1.5B |
| LUFFY-Qwen-Instruct-7B | https://huggingface.co/Elliott/LUFFY-Qwen-Instruct-7B | Qwen2.5-7B-Instruct |

---

