# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: predict
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/12/15 18:02
"""
    文件说明：
            
"""
import torch
from qwen1_8.tokenization_qwen import QWenTokenizer
from qwen1_8.modeling_qwen import QWenLMHeadModel
import argparse


def build_prompt(tokenizer, instruction, text, device):
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return f"{role}\n{content}", tokenizer.encode(role, allowed_special=set()) + nl_tokens + tokenizer.encode(
            content, allowed_special=set())

    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    system_text, system_tokens_part = _tokenize_str("system", "You are a helpful assistant.")
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
    prompt_id = im_start_tokens + _tokenize_str("user", instruction + text)[1] + im_end_tokens
    input_ids = system_tokens + nl_tokens + prompt_id + nl_tokens + tokenizer.encode("assistant", allowed_special=set()) + nl_tokens
    input_ids = torch.tensor([input_ids]).to(device)
    return input_ids


def predict_one_sample(model, tokenizer, instruction, text, args):
    # 获取解码的配置参数，涉及生成内容最大长度、TopP解码的Top-P概率、温度、重复惩罚因子等
    generation_config = model.generation_config
    generation_config.min_length = 5
    generation_config.max_new_tokens = args.max_tgt_len
    generation_config.top_p = args.top_p
    generation_config.temperature = args.temperature
    generation_config.do_sample = args.do_sample
    generation_config.repetition_penalty = args.repetition_penalty
    # 根据文本内容，融合提示词和输入对话内容，构建模型输入所需要的input_ids
    input_ids = build_prompt(tokenizer, instruction, text, model.device)
    # 进行结果预测
    outputs = model.generate(input_ids, generation_config=generation_config,
                             stop_words_ids=[[tokenizer.im_end_id], [tokenizer.im_start_id]])
    # 仅取生成内容
    response = outputs.tolist()[0][len(input_ids[0]):]
    # 将ID内容转化成字符串进行输出
    response = tokenizer.decode(response, skip_special_tokens=True)
    return response


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
    model = QWenLMHeadModel.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                            device_map="cuda:{}".format(args.device))
    model.eval()
    tokenizer = QWenTokenizer.from_pretrained(args.model_path)
    # 内置对话要素抽取的提示词内容
    instruction = "你现在是一个医疗对话要素抽取专家。\n" \
                  "请针对下面对话内容抽取出药品名称、药物类别、医疗检查、医疗操作、现病史、辅助检查、诊断结果和医疗建议等内容，并且以json格式返回，Key为上述待抽取的字段名称，Value为抽取出的文本内容。\n" \
                  "注意事项：（1）药品名称、药物类别、医疗检查和医疗操作的内容会在对话中存在多个，因此Value内容以List形式存放；若抽取内容为空，则为一个空的List；\n" \
                  "（2）抽取出的药品名称、药物类别、医疗检查和医疗操作对应内容，需要完全来自于原文，不可进行修改，内容不重复；\n" \
                  "（3）现病史、辅助检查、诊断结果和医疗建议的内容需要根据整个对话内容进行总结抽取，Value内容以Text形式存放。\n" \
                  "对话文本：\n"
    # 输入对话文本，进行对话要素抽取
    while True:
        print('开始对对话内容进行要素抽取，输入CTRL+C，则退出')
        text = input("输入的对话内容为：")
        response = predict_one_sample(model, tokenizer, instruction, text, args)
        print("对话要素抽取结果为：")
        print(response)
