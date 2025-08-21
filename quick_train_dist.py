import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from verl.utils import hf_tokenizer
from verl.utils.dataset import SFTDataset


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def get_model_tokenizer(model_path, device):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = hf_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    #model.to(device)
    return model, tokenizer

def get_dataloader(data_path, data_config, tokenizer, batch_size, local_rank, world_size):
    if local_rank == 0:
        print(f"Loading data from {data_path} with batch size {batch_size}")
        print(f"Data config: {data_config}")
    dataset = SFTDataset(data_path, tokenizer=tokenizer, config=data_config)
    sampler = DistributedSampler(dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True
    )
    return dataloader


def train():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    model_path = 'Qwen/Qwen2.5-Coder-7B-Instruct'
    model, tokenizer = get_model_tokenizer(model_path, device)
    model = DDP(model, device_ids=[local_rank])

    config_filename = 'sft_trainer.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    data_path = '/mnt/blob-data-sigmasystem/yuhang/system_data/kql_parquet_0725/kql_train_data.parquet'
    train_dataloader = get_dataloader(data_path, data_config, tokenizer, batch_size=6,
                                      local_rank=rank, world_size=world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    num_epochs = 5
    vocab_size = model.module.config.vocab_size
    ckpt_save_path = '/mnt/blob-data-sigmasystem-out/yuhang/system_data/Qwen2.5-7B-Coder-Instruct-Naive-kql-8k-0820'
    os.makedirs(ckpt_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()

        for iter_id, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            shift_labels = batch["labels"][:, 1:].contiguous().view(-1).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=False)
            logits = outputs.logits[..., :-1, :].contiguous().view(-1, vocab_size)

            loss = loss_fct(logits, shift_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_id % 10 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Iter {iter_id}/{len(train_dataloader)} | Loss: {loss.item():.4f}")

        if rank == 0:
            save_path = os.path.join(ckpt_save_path, f"model_epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    train()
