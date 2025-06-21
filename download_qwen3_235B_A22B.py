import os
os.environ['HF_HOME'] = '/mnt/blob-data-sigmasystem-out/yuhang/' 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-235B-A22B"

try:
    print(f"Attempting to download model: {model_name}")

    # Load the tokenizer
    # If HF_HOME is set, it will look there first. Otherwise, it uses the default.
    # If you used cache_directory, pass it here:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    print("Tokenizer loaded successfully.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", # Use "auto" to let transformers decide based on your hardware
        device_map="auto",
        # cache_dir=cache_directory # Uncomment if using Option 2
    )
    print("Model loaded successfully.")
    print(f"Model and tokenizer have been downloaded and/or loaded from cache.")

except Exception as e:
    print(f"An error occurred during download or loading: {e}")
    print("Please check your internet connection, disk space, and Hugging Face authentication (if the model is gated).")
