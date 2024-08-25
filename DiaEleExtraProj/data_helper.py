# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/12/15 18:02
"""
    文件说明：
            
"""
import json
import random


def data_helper(path, train_save_path, test_save_path):
    """
    数据处理函数
    Args:
        path: 原始数据文件路径
        train_save_path: 训练数据文件路径
        test_save_path: 测试数据文件路径

    Returns:

    """
    save_data = []
    # 加载处理对话数据
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        # 遍历所有数据
        for i, line in enumerate(data):
            # 创建提示词，根据角色、任务、详细要求三个方面编写提示词内容
            instruction = "你现在是一个医疗对话要素抽取专家。\n" \
                          "请针对下面对话内容抽取出药品名称、药物类别、医疗检查、医疗操作、现病史、辅助检查、诊断结果和医疗建议等内容，并且以json格式返回，Key为上述待抽取的字段名称，Value为抽取出的文本内容。\n" \
                          "注意事项：（1）药品名称、药物类别、医疗检查和医疗操作的内容会在对话中存在多个，因此Value内容以List形式存放；若抽取内容为空，则为一个空的List；\n" \
                          "（2）抽取出的药品名称、药物类别、医疗检查和医疗操作对应内容，需要完全来自于原文，不可进行修改，内容不重复；\n" \
                          "（3）现病史、辅助检查、诊断结果和医疗建议的内容需要根据整个对话内容进行总结抽取，Value内容以Text形式存放。\n" \
                          "对话文本：\n"
            # 将数据按照instruction、input和output形式进行保存，并将输出结果修改markdown形式，方便后面解析
            sample = {"instruction": instruction, "input": line["dialogue_text"],
                      "output": "```json\n{}\n```".format(line["extract_text"]).replace("\'", "\"")}
            save_data.append(sample)
    # 将数据随机打乱
    random.shuffle(save_data)

    # 遍历数据，将其分别保存到训练文件和测试文件中
    fin_train = open(train_save_path, "w", encoding="utf-8")
    fin_test = open(test_save_path, "w", encoding="utf-8")
    for i, sample in enumerate(save_data):
        if i < 50:
            fin_test.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            fin_train.write(json.dumps(sample, ensure_ascii=False) + "\n")
    fin_train.close()
    fin_test.close()


if __name__ == '__main__':
    path = "data/all.json"
    train_save_path = "data/train.json"
    test_save_path = "data/test.json"
    data_helper(path, train_save_path, test_save_path)
