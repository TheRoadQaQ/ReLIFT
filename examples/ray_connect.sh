export MASTER_ADDR=""
export NUMEXPR_MAX_THREADS=100
export VLLM_ATTENTION_BACKEND=XFORMERS
ray stop
ray start --address=$MASTER_ADDR:6379 --num-cpus=100
