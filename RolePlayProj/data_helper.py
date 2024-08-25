# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2024/1/31 21:14
"""
    文件说明：
            
"""
import json
import random


def merge_data(path_list, save_path):
    """
    数据合并函数
    Args:
        path_list: 待合并的文件路径
        save_path: 保存的文件路径

    Returns:

    """
    fin = open(save_path, "w", encoding="utf-8")
    # 遍历所有待合并文件
    for path in path_list:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                # 利用json.loads加载数据
                sample = json.loads(line.strip())
                # 对于角色背景知识数据，去除对话数据中前置的角色文本信息
                if "specific" in path:
                    sample["question"] = "".join(sample["question"].split("，")[1:])
                # 过滤非"孙悟空"、"张飞"、"李白"的角色数据
                if sample["role"] not in ["孙悟空", "张飞", "李白"]:
                    continue
                # 将数据写入到保存文件中
                fin.write(json.dumps(sample, ensure_ascii=False) + "\n")
    fin.close()


def data_helper(path, desc_path, save_path):
    """
    数据预处理函数
    Args:
        path: 数据路径
        desc_path: 角色描述数据路径
        save_path: 保存路径

    Returns:

    """
    # 加载角色描述数据
    with open(desc_path, "r", encoding="utf-8") as fh:
        desc_data = json.load(fh)
    fin = open(save_path, "w", encoding="utf-8")
    # 定义系统提示词
    instruction = "你是{}，你的特征描述是：{}。现在请你回答我的一些问题，以准确展现你的人格特征！你的说话风格要全面模仿被赋予的人格角色！请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要嗦，也不要太过于正式或礼貌。"
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            # 利用json.loads加载数据
            line = json.loads(line.strip())
            # 对多个回复内容进行随机打乱
            random.shuffle(line["generated"])
            # 如果有多个回复内容，最多取两个
            for i in range(min(2, len(line["generated"]))):
                # 按照指令、输入、输出的格式保存数据
                sample = {"instruction": instruction.format(line["role"], desc_data[line["role"]]),
                          "input": line["question"],
                          "output": line["generated"][i]}
                fin.write(json.dumps(sample, ensure_ascii=False) + "\n")
    fin.close()


if __name__ == '__main__':
    merge_data(["data/rolebench-zh_role_specific_train.jsonl", "data/rolebench-zh_general_train.jsonl"],
               "data/role_train.jsonl")
    merge_data(["data/rolebench-zh_role_specific_test.jsonl", "data/rolebench-zh_general_test.jsonl"],
               "data/role_test.jsonl")
    data_helper("data/role_train.jsonl", "data/profiles-zh_desc.json", "data/train.json")
    data_helper("data/role_test.jsonl", "data/profiles-zh_desc.json", "data/test.json")
