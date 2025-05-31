MODEL_NAME=Qwen/Qwen3-32B
export HF_HOME=/mnt/blob-data-sigmasystem/yuhang
train_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_train_data.parquet
test_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_test_data.parquet
ckpt_dir=/mnt/blob-data-sigmasystem-out/yuhang/system_data/ckpt_4nodes_system_qwen3_32B

torchrun --nnodes=4  --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
	data.train_files=$train_file \
        data.val_files=$test_file \
	data.prompt_key=prompt \
        data.response_key=answer \
	data.train_batch_size=64 \
        data.micro_batch_size_per_gpu=2 \
	data.truncation=right \
	data.max_length=4096 \
	model.partial_pretrain=$MODEL_NAME \
        trainer.project_name=system-sft \
	trainer.experiment_name=system-qwen3_32B \
        trainer.total_epochs=50 \
	trainer.default_local_dir=$ckpt_dir \
        trainer.logger=['console'] \
	trainer.default_hdfs_dir=null
