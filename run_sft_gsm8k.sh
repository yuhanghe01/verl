MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
#train_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_train_data.parquet
#test_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_test_data.parquet
ckpt_dir=/mnt/blob-data-sigmasystem-out/yuhang/gsm8k_data_Qwen_0.5B

python \
    -m verl.trainer.main_sft \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=16 \
    data.max_length=2000 \
    trainer.project_name=system-sft \
    trainer.experiment_name=system-sft \
    trainer.total_epochs=10 \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['console','wandb'] \
    trainer.ckpt_save_dir=$ckpt_dir \
    trainer.default_hdfs_dir=null
