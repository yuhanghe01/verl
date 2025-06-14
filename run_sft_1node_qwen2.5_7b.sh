MODEL_NAME=Qwen/Qwen2.5-Coder-7B 
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-Coder-7B')"

train_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_train_data.parquet
test_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_test_data.parquet
ckpt_dir=/mnt/blob-data-sigmasystem-out/yuhang/system_data_Qwen2.5-7B-Codeir-Naive

python \
    -m verl.trainer.main_sft_torchtrainer \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=3 \
    data.max_length=4096 \
    optim.lr=5e-5 \
    model.model_path=$MODEL_NAME \
    trainer.project_name=sft-coder-7b-0613 \
    trainer.experiment_name=sft-coder-7b-0613 \
    trainer.total_epochs=5 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['wandb','console'] \
    trainer.ckpt_save_dir=$ckpt_dir \
    trainer.default_hdfs_dir=null
