# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: predict
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2024/1/31 22:51
"""
    文件说明：
            
"""
import torch
from baichuan.tokenization_baichuan import BaichuanTokenizer
from baichuan.modeling_baichuan import BaichuanForCausalLM
import argparse
from transformers.generation.utils import GenerationConfig
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1", help="")
    parser.add_argument("--model_path", type=str, default="output_dir_qlora/epoch-3-step-906-merge/", help="")
    parser.add_argument("--max_tgt_len", type=int, default=512, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--top_p", type=float, default=0.8, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="")
    return parser.parse_args()


if __name__ == '__main__':
    # 设置预测的配置参数
    args = parse_args()
    # 加载融合Lora参数后的模型以及Tokenizer
    tokenizer = BaichuanTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = BaichuanForCausalLM.from_pretrained(args.model_path, device_map="cuda:{}".format(args.device),
                                                torch_dtype=torch.bfloat16)
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(args.model_path)
    with open(args.desc_path, "r", encoding="utf-8") as fh:
        desc_data = json.load(fh)

    # 输入对话文本，进行对话要素抽取
    while True:
        print('开始对对话内容进行要素抽取，输入CTRL+C，则退出')
        role = input("目前支持3种角色（张飞、李白、孙悟空）的对话，请输入角色名称：")
        if role not in ["张飞", "李白", "孙悟空"]:
            print("角色名称错误，请重新输入")
            continue
        text = input("用户：")
        messages = []
        messages.append({"role": "system",
                         "content": "你是{}，你的特征描述是：{}。现在请你回答我的一些问题，以准确展现你的人格特征！你的说话风格要全面模仿被赋予的人格角色！请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要嗦，也不要太过于正式或礼貌。".format(role, desc_data[role])})
        messages.append({"role": "user", "content": text})
        response = model.chat(tokenizer, messages)
        print("回答：", response)
