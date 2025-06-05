# -*- coding: utf-8 -*-
import os
import hydra
import ray
from verl.trainer.sft.ray_trainer import RaySFTTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.sft_fsdp_workers import SFTLMWorker
from verl.trainer.sft.ray_trainer import ResourcePoolManager, Role
from omegaconf import OmegaConf
from pprint import pprint

@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)

def run_sft(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", 
                                      "NCCL_DEBUG": "WARN", 
                                      "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)
        role_worker_mapping = {Role.FSDP_WORKER: ray.remote(SFTLMWorker)}

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.FSDP_WORKER: global_pool_id}

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, 
                                                    mapping=mapping)
        ray_worker_group_cls = RayWorkerGroup
        trainer = RaySFTTrainer(
            config=config,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )
        trainer.init_workers()
        trainer.fit()

if __name__ == "__main__":
    main()