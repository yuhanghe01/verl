MODEL_NAME=Qwen/Qwen3-30B-A3B 
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-Coder-32B')"

train_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_train_data.parquet
test_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_test_data.parquet
ckpt_dir=/mnt/blob-data-sigmasystem-out/yuhang/system_data_Qwen3-30B-A3B

python \
    -m verl.trainer.main_sft \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=2 \
    data.max_length=4096 \
    optim.lr=1e-5 \
    model.model_path=$MODEL_NAME \
    model.lora_rank=32 \
    trainer.project_name=system-sft-qwen3-30b-a3b \
    trainer.experiment_name=system-sft-qwen3-30b-a3b \
    trainer.total_epochs=10 \
    trainer.nnodes=8 \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['wandb','console'] \
    trainer.ckpt_save_dir=$ckpt_dir \
    trainer.default_hdfs_dir=null
