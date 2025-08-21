MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
#python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-Coder-7B-Instruct')"
torchrun --nnodes=1 --nproc_per_node=8 quick_train_dist.py