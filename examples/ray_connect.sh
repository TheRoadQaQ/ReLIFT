export MASTER_ADDR="?"
export NUMEXPR_MAX_THREADS=100
ray stop
ray start --address=$MASTER_ADDR:6379 --num-cpus=100
