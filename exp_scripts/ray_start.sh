export MASTER_ADDR=30.207.97.15
export NUMEXPR_MAX_THREADS=100
ray stop
ray start --head --num-cpus=100 --node-ip-address=30.207.97.15 --port=6379
