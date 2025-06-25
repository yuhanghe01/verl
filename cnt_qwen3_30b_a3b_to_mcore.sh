set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

python examples/data_preprocess/gsm8k.py
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

mcore_save_path=/mnt/blob-data-sigmasystem-out/yuhang/mcore-Qwen-Qwen3-30B-A3
mkdir -p $mcore_save_path

python scripts/converter_hf_to_mcore.py --hf_model_path=/mnt/blob-data-sigmasystem/yuhang/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9 output_path=$mcore_save_path
