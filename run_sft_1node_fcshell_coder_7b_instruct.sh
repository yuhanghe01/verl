MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct 
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-Coder-7B-Instruct')"

train_file=/mnt/blob-data-sigmasystem/yuhang/system_data/fcshell_parquet_0619/fcshell_train_data.parquet
test_file=/mnt/blob-data-sigmasystem/yuhang/system_data/fcshell_parquet_0619/fcshell_test_data.parquet
ckpt_dir=/mnt/blob-data-sigmasystem-out/yuhang/system_data/Qwen2.5-7B-Coder-Instruct-Naive-fcshell

python \
    -m verl.trainer.main_sft_torchtrainer \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=3 \
    data.max_length=4096 \
    data.truncation=left \
    optim.lr=1e-5 \
    model.model_path=$MODEL_NAME \
    trainer.project_name=fcshell-coder-7b-instruct-0619 \
    trainer.experiment_name=fcshell-coder-7b-instruct-0619 \
    trainer.total_epochs=5 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['wandb','console'] \
    trainer.ckpt_save_dir=$ckpt_dir \
    trainer.default_hdfs_dir=null
