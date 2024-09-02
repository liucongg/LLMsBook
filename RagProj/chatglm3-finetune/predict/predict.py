# -*- coding: utf-8 -*-


import argparse
from transformers import AutoTokenizer, AutoModel


def get_result(model_path, question):
    """

    :param model_path:  模型路径
    :param question: 问题
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model = model.eval()
    input_ = question
    response, history = model.chat(tokenizer, input_, history=[])
    print(response)

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM-6B inference test.')
    parser.add_argument('--model_path', type=str, required=True, help='待测试的模型保存路径')
    parser.add_argument('--instruction_text', type=str, default='写1000字的文章：\n', help='测试输入的指令')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    question = "写一个制作披萨的步骤指南。"
    get_result(model_path=args.model_path,question=question)





