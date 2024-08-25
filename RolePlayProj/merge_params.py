# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: merge_params
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2024/1/31 21:15
"""
    文件说明：
            
"""
import torch
from baichuan.modeling_baichuan import BaichuanForCausalLM
from baichuan.tokenization_baichuan import BaichuanTokenizer
import argparse
from peft import PeftModel


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--ori_model_dir', default="Baichuan2-7B-Chat/",
                        type=str, help='')
    parser.add_argument('--model_dir',
                        default="output_dir/epoch-1-step-1002",
                        type=str, help='')
    parser.add_argument('--save_model_dir',
                        default="output_dir/epoch-1-step-1002-merge/",
                        type=str, help='')
    return parser.parse_args()


def main():
    # 设置模型融合参数
    args = set_args()
    if args.device == "-1":
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    # 加载百川2原始模型
    base_model = BaichuanForCausalLM.from_pretrained(args.ori_model_dir, torch_dtype=torch.float16, device_map=device)
    tokenizer = BaichuanTokenizer.from_pretrained(args.ori_model_dir)
    # 加载Lora外挂参数
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    # 将外挂参数合并到原始参数中
    model = lora_model.merge_and_unload()
    # 将合并后的参数进行保存
    model.save_pretrained(args.save_model_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(args.save_model_dir)


if __name__ == '__main__':
    main()
