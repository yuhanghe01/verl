from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from verl.utils import hf_processor, hf_tokenizer
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset import SFTDataset
import yaml
import torch.nn as nn
import os


def get_dataloader(data_path, data_config, tokenizer):
    dataset = SFTDataset(data_path, data_config, tokenizer)
    
    train_dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=12,
        drop_last=True,
        collate_fn=None,
        shuffle=True,
    )

    return train_dataloader


def get_model_tokenizer():
    model_path = 'Qwen/Qwen2.5-Coder-7B-Instruct'
    model_config = AutoConfig.from_pretrained(model_path, 
                                              trust_remote_code=True)
        
    tokenizer = hf_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        attn_implementation="flash_attention_2",
        device_map="auto",  # handled by FSDP accelerate launch
        torch_dtype="auto",  # use float16 for inference
        trust_remote_code=True
    )
    model.train()
    
    return model, tokenizer

def train():
    model, tokenizer = get_model_tokenizer()
    
    # Load the dataset
    config_filename = 'sft_trainer_fcshell.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # data_path = ['/mnt/yuhang/SFT/kql_parquet_0626/kql_train_data.parquet']
    data_path = ['/mnt/blob-data-sigmasystem/yuhang/system_data/fcshell_parquet_0825/fcshell_train_data.parquet']
    train_dataloader = get_dataloader(data_path, tokenizer, data_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    num_epochs = 10
    vocab_size = model.config.vocab_size #152064
    ckpt_save_path = '/mnt/blob-data-sigmasystem-out/yuhang/system_data/FCShell_Qwen_7B_ckpt_appsysprompt'
    os.makedirs(ckpt_save_path, exist_ok=True)
    
    for epoch in range(num_epochs):
        for iter_id, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            position_ids = batch["position_ids"].to("cuda")
            # loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(device_name)
            loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

            # labels = input_ids[:, 1:].contiguous()
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
            logits = output.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, vocab_size)
            
            shift_labels = batch["labels"][:, 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = loss_fct(shift_logits, shift_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_id % 10 == 0:
                # Print loss every 10 iterations
                print(f"Loss: {loss.item()}, Epoch: {epoch + 1}/{num_epochs}, Iteration: {iter_id}/{len(train_dataloader)}")
            
        # Save the model checkpoint
        save_path = os.path.join(ckpt_save_path, f"model_epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        
    print('Done!')

if __name__ == "__main__":
    train()
    # model, tokenizer = get_model_tokenizer()
    # print(model)
    # print(tokenizer)
    # breakpoint()
