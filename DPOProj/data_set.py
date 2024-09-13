# -*- coding:utf-8 -*-
# @project: AIGC
# @filename: data_set
# @author: 杜振东.py
# @contact: zddu@iyunwen.com
# @time: 2023/12/28 19:16
"""
    文件说明：数据集加载
            
"""
from datasets import load_dataset
def load_dataset(dataset_name="liyucheng/zhihu_rlhf_3k",ratio=0.1):
    '''
    加载偏好对齐数据集
    数据集链接：https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k
    Args:
        dataset_name: 数据集名称【注意务必是datasets支持的数据集格式】
        ratio: 验证集占总数据集比例

    Returns:
        train_data: 偏好对齐训练样本
        dev_data: 偏好对齐验证样本
    '''
    data_zh = load_dataset(path=dataset_name)
    data_all = data_zh['train'].train_test_split(ratio)
    train_data = data_all['train']
    dev_data = data_all['test']
    print(len(train_data))
    print(len(dev_data))
    return train_data,dev_data