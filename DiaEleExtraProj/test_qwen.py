# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: test_qwen
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/12/23 21:02
"""
    文件说明：
            
"""
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict_openai(model, instruction, text):
    """
    利用Qwen模型进行对话要素抽取实战
    Args:
        model: Qwen模型
        instruction: 提示词内容
        text: 对话内容

    Returns:

    """
    response, history = model.chat(tokenizer, instruction + text, history=None)
    return response


if __name__ == '__main__':
    # 实例化Qwen-1.8B模型以及Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen-1_8-chat/", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen-1_8-chat/", device_map="cuda:1", trust_remote_code=True).eval()
    print('开始对对话内容进行要素抽取，输入CTRL+C，则退出')
    while True:
        # 输入提示词内容和对话内容
        instruction = input("输入的提示词内容为：")
        text = input("输入的对话内容为：")
        # 进行对话要素抽取
        response = predict_openai(model, instruction, text)
        print("对话要素抽取结果为：")
        print(response)
