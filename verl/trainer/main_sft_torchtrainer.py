# -*- coding: utf-8 -*-
import os
import hydra
import ray
from ray.train import get_context
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from verl.trainer.sft.ray_trainer import RaySFTTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.sft_fsdp_workers_naive import SFTLMWorker
from verl.trainer.sft.ray_trainer import ResourcePoolManager, Role
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pprint import pprint
import torch
import torch.distributed as dist
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def train_loop_per_worker(config):
    ctx = get_context()
    local_rank = ctx.get_local_rank()
    world_size = ctx.get_world_size()
    rank = ctx.get_world_rank()
    
    torch.cuda.set_device(local_rank)
    
    sft_trainer = SFTLMWorker(config)
    
    result = sft_trainer.fit()
    
    return result
    

@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(cfg: DictConfig):
    if not ray.is_initialized():
        ray.init()
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(
            num_workers=cfg.trainer.nnodes * cfg.trainer.n_gpus_per_node,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
            trainer_resources={"CPU": 1},
        ),
        train_loop_config=cfg,
    )
    trainer.fit()
    pprint(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
