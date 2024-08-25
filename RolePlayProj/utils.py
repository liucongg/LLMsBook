# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: utils
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2024/1/31 21:15
"""
    文件说明：
            
"""
import torch
import random
import numpy as np
from transformers import set_seed
import json
import os
from torch.utils.data import Dataset
import bitsandbytes as bnb
from tqdm import tqdm
import math


class Baichuan2PromptDataSet(Dataset):
    """角色扮演所需的数据类"""

    def __init__(self, data_path, tokenizer, max_len, max_src_len, generation_config, is_skip):
        """
        初始化函数
        Args:
            data_path: 文件数据路径
            tokenizer: 分词器
            max_len:  模型训练最大长度
            max_src_len: 模型输入最大长度
            generation_config: 模型相关配置信息
            is_skip: 不符合长度标准数据是否跳过
        """
        self.all_data = []
        skip_data_number = 0
        # 遍历文件中的每一个样本
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False
                # 构造角色扮演模型所需系统指令内容
                sys_prompt_id = tokenizer.encode(sample["instruction"])
                # 构建用户输入内容
                prompt_id = tokenizer.encode(sample["input"])
                # 当用户输入内容长度超过最大长度时，进行向前截断，并生成对应的label
                if len(prompt_id) > max_src_len:
                    prompt_id = prompt_id[:max_src_len]
                    skip_flag = True
                input_ids = [generation_config.user_token_id] + prompt_id + [generation_config.assistant_token_id]
                labels = [-100] * (len(prompt_id) + 2)
                # 构建模型输出内容
                output_id = tokenizer.encode(sample["output"])
                # 当模型输出内容长度超过最大长度时，进行向前截断
                max_tgt_len = max_len - 1 - len(input_ids) - len(sys_prompt_id)
                if len(output_id) > max_tgt_len:
                    output_id = output_id[:max_tgt_len]
                    skip_flag = True
                # 将系统指令、用户输入、模型输出进行拼接，构建完整的模型训练所需数据
                input_ids = sys_prompt_id + input_ids + output_id + [tokenizer.eos_token_id]
                labels = [-100] * len(sys_prompt_id) + labels + output_id + [tokenizer.eos_token_id]
                assert len(input_ids) <= max_len
                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                # 将每个样本进行保存，用于后续训练使用
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


class DataCollator(object):
    """DataLoader所需DataCollator类"""

    def __init__(self, tokenizer):
        """
        初始化函数
        Args:
            tokenizer: 分词器
        """
        self.tokenizer = tokenizer
        # padding标记
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        """
        回调函数
        Args:
            batch: 输入数据

        Returns:

        """
        # 获取一个Batch内的最大长度
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = math.ceil(max(lengths) / 8) * 8
        # 遍历Batch内容
        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]
            # 将Batch内的数据填充到最大长度
            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
        # 将结果转化成torch.tensor并返回
        return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long)}


def print_trainable_parameters(model):
    """打印可训练参数"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {} || all params: {} || trainable%: {}".format(trainable_params, all_param,
                                                                            100 * trainable_params / all_param))


def print_rank_0(msg, rank=0):
    """多卡训练时，打印rank=0上的信息"""
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    """将Batch内的数据内容，设置到对应的device上"""
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def set_random_seed(seed):
    """设置随机种子，方便模型进行复现"""
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_model(model, tokenizer, output_dir, model_name):
    """模型保存，保存模型和对应的分词器"""
    save_dir = os.path.join(output_dir, model_name)
    model.save_pretrained(save_dir, torch_dtype=torch.float16)
    tokenizer.save_pretrained(save_dir)


def find_all_linear_names(model):
    """找到模型中的所有线性层"""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def evaluation(model, eval_dataloader, device):
    """模型验证，计算验证集的PPL值"""
    model.eval()
    total_loss = 0
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), unit="batch"):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            total_loss += loss.float()
    total_loss = total_loss / (step + 1)

    try:
        perplexity = torch.exp(total_loss)
    except OverflowError:
        perplexity = float("inf")
    try:
        perplexity = get_all_reduce_mean(perplexity).item()
    except:
        pass
    model.train()
    return perplexity


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor
