#!/bin/bash

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results/

# 定义模型路径、名称、模板的数组
MODEL_PATHS=(
    "/jizhicfs/hymiezhao/ml/reasoning/ReLIFT/train_results/ReLIFT/math_7b_relift_test/best/actor"
)
MODEL_NAMES=(
    "relift-7B"
)
TEMPLATES=(
    "own"
)

export VLLM_ATTENTION_BACKEND=XFORMERS

ray stop
ray start --head --num-cpus=100

# 遍历所有模型
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    TEMPLATE=${TEMPLATES[$i]}

    echo "Running inference for $MODEL_NAME ..."
  
    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
      --model_path "$MODEL_PATH" \
      --input_file "$DATA" \
      --remove_system True \
      --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
      --template "$TEMPLATE"
done