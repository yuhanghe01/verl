# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """
    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        # prompt_key = config.get("prompt_key", "prompt")
        # # prompt_dict_keys = config.get("prompt_dict_keys", None)
        # # response_key = config.get("response_key", "response")
        # response_dict_keys = config.get("response_dict_keys", None)
        # max_length = config.get("max_length", 1024)
        # truncation = config.get("truncation", "error")

        # assert truncation in ["error", "left", "right"]
        # self.truncation = truncation
        
        self.config = config

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        # self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        # # self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        # # self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        # self.response_dict_keys = response_dict_keys if response_dict_keys else []

        # self.max_length = max_length

        # self._download()
        self.read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        
        all_prompts = self.dataframe[self.config['prompt_key']]
        all_responses = self.dataframe[self.config['response_key']]
        
        self.prompts, self.responses = list(), list()
        
        for prompt_tmp, response_tmp in zip(all_prompts, all_responses):
            if isinstance(prompt_tmp, (pd.core.series.Series, np.ndarray)):
                prompt_tmp = prompt_tmp.tolist()
            if isinstance(response_tmp, (pd.core.series.Series, np.ndarray)):
                response_tmp = response_tmp.tolist()
            
            self.prompts.append(prompt_tmp)
            self.responses.append(response_tmp['answer'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt = self.prompts[item]
        response = self.responses[item]
        prompt = sorted(prompt, key=lambda x: 0 if x['role'] == "system" else 1)
        
        prompt.append({'role': 'assistant', 'content': response})
        prompt_chat_str = self.tokenizer.apply_chat_template(prompt, 
                                                        add_generation_prompt=False, 
                                                        tokenize=False)
        prompt_ids_output = self.tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        attention_mask = prompt_ids_output["attention_mask"][0]
        
        assistant_token_ids = self.tokenizer("<|im_start|>assistant\n")['input_ids']
        last_im_start_id = np.where(prompt_ids == assistant_token_ids[0])[0][-1]
        
        if prompt_ids[last_im_start_id + 1] == assistant_token_ids[1] and \
           prompt_ids[last_im_start_id + 2] == assistant_token_ids[2]:
            # the last im_start is assistant
            assistant_start_id = last_im_start_id + 3
        else:
            # the last im_start is user
            raise ValueError(f"Last im_start is not assistant, {prompt_ids[last_im_start_id:last_im_start_id + 3]}")
        
        labels = prompt_ids.clone()
        labels[:assistant_start_id] = -100  # ignore system, user prompt in loss calculation

        # padding to max length
        sequence_length = prompt_ids.shape[0]
        # print(f"sequence_length: {sequence_length}, max_length: {self.config['max_length']}")
        if sequence_length < self.config['max_length']:
            padded_input_ids = torch.ones(size=(self.config['max_length'] - sequence_length,), dtype=prompt_ids.dtype) * self.tokenizer.pad_token_id
            padded_labels = torch.ones(size=(self.config['max_length'] - sequence_length,), dtype=labels.dtype) * -100
            padded_mask = torch.zeros(size=(self.config['max_length'] - sequence_length,), dtype=attention_mask.dtype)
            prompt_ids = torch.cat((prompt_ids, padded_input_ids))
            labels = torch.cat((labels, padded_labels))
            attention_mask = torch.cat((attention_mask, padded_mask))
        elif sequence_length > self.config['max_length']:
            if self.config['truncation'] == "left":
                prompt_ids = prompt_ids[-self.config['max_length'] :]
                labels = labels[-self.config['max_length'] :]
                attention_mask = attention_mask[-self.config['max_length'] :]
            elif self.config['truncation']  == "right":
                prompt_ids = prompt_ids[: self.config['max_length']]
                labels = labels[: self.config['max_length']]
                attention_mask = attention_mask[: self.config['max_length']]
            elif self.config['truncation']  == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.config['max_length']=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.config['truncation']}")

        position_ids = compute_position_id_with_mask(attention_mask)*attention_mask
        
        return {
            "input_ids": prompt_ids,
            "position_ids": position_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        

def test():
    from verl.utils import hf_processor, hf_tokenizer
    import yaml
    config_filename = 'verl/trainer/config/sft_trainer.yaml'
    
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
        
    model_path = 'Qwen/Qwen2.5-Coder-7B-Instruct'
    tokenizer = hf_tokenizer(model_path)
    
    sft_data = SFTDataset(
        parquet_files='/mnt/yuhang/SFT/fcshell_parquet_0626/fcshell_train_data.parquet',
        tokenizer=tokenizer,
        config=config['data']
    )
    
    data_loader = torch.utils.data.DataLoader(
        sft_data,
        batch_size=2,
    )

    # max_length = 0
    for batch in data_loader:
        print(batch['input_ids'].shape)
        # print(batch['attention_mask'].shape)
        print(batch['position_ids'].shape)
        # print(batch['loss_mask'].shape)
        print(batch['labels'].shape)
        # breakpoint()
    
    
# test()
