# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: merge_params
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/12/16 22:33
"""
    文件说明：
            
"""
import torch
from qwen1_8.modeling_qwen import QWenLMHeadModel
from qwen1_8.tokenization_qwen import QWenTokenizer
import argparse
from peft import PeftModel


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--ori_model_dir', default="Qwen-1_8-chat/",
                        type=str, help='')
    parser.add_argument('--model_dir', default="output_dir_qlora/epoch-3-step-906/",
                        type=str, help='')
    parser.add_argument('--save_model_dir', default="output_dir_qlora/epoch-3-step-906-merge/",
                        type=str, help='')
    return parser.parse_args()


def main():
    # 设置模型融合参数
    args = set_args()
    if args.device == "-1":
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    # 加载千问原始模型
    base_model = QWenLMHeadModel.from_pretrained(args.ori_model_dir, torch_dtype=torch.float16, device_map=device)
    tokenizer = QWenTokenizer.from_pretrained(args.ori_model_dir)
    # 加载Lora外挂参数
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    # 将外挂参数合并到原始参数中
    model = lora_model.merge_and_unload()
    # 将合并后的参数进行保存
    model.save_pretrained(args.save_model_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(args.save_model_dir)


if __name__ == '__main__':
    main()
