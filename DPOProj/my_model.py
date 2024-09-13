# -*- coding:utf-8 -*-
# @project: AIGC
# @filename: my_model
# @author: 杜振东.py
# @contact: zddu@iyunwen.com
# @time: 2023/12/29 20:23
"""
    文件说明：模型读取
            
"""
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled
import torch

def load_model(model_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"):
    '''
    加载模型
    模型链接：https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6
    Args:
        model_path: 模型路径【注意务必是HF支持的路径格式格式】

    Returns:
        model: 加载好的模型
        tokenizer: 与之相对应的分词器
        ref_model: 参考模型
    '''
    config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            cache_dir=None
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        load_in_4bit=True,
        load_in_8bit=False,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map='auto',
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        ),
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        load_in_8bit=False,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map='auto',
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bt_use_double_quant=True,
            bnb_4biti_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    return model, tokenizer, ref_model