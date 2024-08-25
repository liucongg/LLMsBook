# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: test_openai
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/12/23 19:52
"""
    文件说明：
            
"""

from openai import OpenAI
import os


def predict_openai(model, instruction, text):
    """
    利用OpenAI的gpt-3.5接口进行对话要素抽取实战
    Args:
        model: OpenAI实例类
        instruction: 提示词内容
        text: 对话内容

    Returns:

    """
    response = model.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction + text},
        ]
    )
    result = response.choices[0].message.content
    return result


if __name__ == '__main__':
    # 设置OpenAI的Key
    os.environ["OPENAI_API_KEY"] = ""
    # 实例化OpenAI类，用于接口调用
    model = OpenAI()
    print('开始对对话内容进行要素抽取，输入CTRL+C，则退出')
    while True:
        # 输入提示词内容和对话内容
        instruction = input("输入的提示词内容为：")
        text = input("输入的对话内容为：")
        # 进行对话要素抽取
        response = predict_openai(model, instruction, text)
        print("对话要素抽取结果为：")
        print(response)
