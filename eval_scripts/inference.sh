#!/bin/bash

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results/

# 定义模型路径、名称、模板的数组
MODEL_PATHS=(
    "RoadQAQ/ReLIFT-Qwen2.5-7B-Zero",
    "RoadQAQ/ReLIFT-Qwen2.5-Math-1.5B-Zero",
    "RoadQAQ/ReLIFT-Qwen2.5-Math-7B-Zero",
)
MODEL_NAMES=(
    "relift-7B"
    "relift-math-1.5B"
    "relift-math-7b"
)
TEMPLATES=(
    "own"
    "own"
    "own"
)

# 遍历所有模型
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    TEMPLATE=${TEMPLATES[$i]}

    echo "Running inference for $MODEL_NAME ..."

    CUDA_VISIBLE_DEVICES=0 python generate_vllm.py \
      --model_path "$MODEL_PATH" \
      --input_file "$DATA" \
      --remove_system True \
      --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
      --template "$TEMPLATE"
done


