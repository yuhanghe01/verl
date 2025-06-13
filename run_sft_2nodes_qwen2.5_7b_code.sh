MODEL_NAME=Qwen/Qwen2.5-Coder-7B 
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-Coder-7B')"

train_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_train_data.parquet
test_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_test_data.parquet
ckpt_dir=/mnt/blob-data-sigmasystem-out/yuhang/system_data_Qwen2.5-Coder-7B-Naive

python \
    -m verl.trainer.main_sft_torchtrainer \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=8 \
    data.max_length=4096 \
    model.model_path=$MODEL_NAME \
    trainer.project_name=system-sft-qwen2.5-coder-7b-naive \
    trainer.experiment_name=system-sft-qwen2.5-coder-7b-naive \
    trainer.total_epochs=10 \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['wandb','console'] \
    trainer.ckpt_save_dir=$ckpt_dir \
    trainer.default_hdfs_dir=null
