MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
python -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

train_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_train_data.parquet
test_file=/mnt/blob-data-sigmasystem/yuhang/system_data/system_test_data.parquet
ckpt_dir=/mnt/blob-data-sigmasystem-out/yuhang/system_data_Qwen_0.5B


# torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#     -m verl.trainer.fsdp_sft_trainer\
#     data.train_files=/datadisk/SFT/fcshell_train_data.parquet \
#     data.val_files=/datadisk/SFT/fcshell_test_data.parquet \
#     data.prompt_key=prompt \
#     data.response_key=answer \
#     data.train_batch_size=256 \
#     data.micro_batch_size_per_gpu=16 \
#     model.partial_pretrain=$MODEL_NAME \
#     trainer.project_name=gsm8k-sft \
#     trainer.experiment_name=gsm8k-sft \
#     trainer.total_epochs=10 \
#     trainer.logger=['console'] \
#     trainer.default_local_dir=/datadisk/SFT/gsm8k_SFT \
#     trainer.default_hdfs_dir=null $@

python \
    -m verl.trainer.main_sft \
    data.train_files=$train_file \
    data.val_files=$test_file \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=512 \
    data.max_length=4000 \
    trainer.project_name=system-sft \
    trainer.experiment_name=system-sft \
    trainer.total_epochs=10 \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['console'] \
    trainer.ckpt_save_dir=/datadisk/SFT/gsm8k_SFT \
    trainer.default_hdfs_dir=null
# torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#     -m verl.trainer.main_sft \
#     data.train_files=/datadisk/SFT/fcshell_train_data.parquet \
#     data.val_files=/datadisk/SFT/fcshell_test_data.parquet \
#     data.prompt_key=prompt \
#     data.response_key=answer \
#     data.train_batch_size=4 \
#     data.micro_batch_size_per_gpu=4 \
#     trainer.project_name=gsm8k-sft \
#     trainer.experiment_name=gsm8k-sft \
#     trainer.total_epochs=10 \
#     trainer.logger=['console'] \
#     trainer.default_local_dir=/datadisk/SFT/gsm8k_SFT \
#     trainer.default_hdfs_dir=null $@

# torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#     -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=/home/yuhanghe/gsm8k/train.parquet \
#     data.val_files=/home/yuhanghe/gsm8k/test.parquet \
#     data.prompt_key=prompt \
#     data.response_key=answer \
#     data.micro_batch_size_per_gpu=4 \
#     model.partial_pretrain=$MODEL_NAME \
#     trainer.project_name=gsm8k-sft \
#     trainer.experiment_name=gsm8k-sft-gemma-1.1-7b-it \
#     trainer.total_epochs=4 \
#     trainer.logger=['console'] \
#     trainer.default_hdfs_dir=null $@
