export MASTER_ADDR=""
export NUMEXPR_MAX_THREADS=100
export VLLM_ATTENTION_BACKEND=XFORMERS
ray stop
ray start --head --num-cpus=100 --node-ip-address=$MASTER_ADDR --port=6379
