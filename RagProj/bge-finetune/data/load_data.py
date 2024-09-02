import os
import random
import argparse
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser(description='data helper')
    parser.add_argument('--data_path', type=str, help='Multi-Doc-QA-Chinese/raw的地址')
    parser.add_argument('--save_home', type=str, default="./", help='训练及验证数据保存路径')
    parser.add_argument('--ratio', type=float, default=0.99, help='训练集占比')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def get_data(home, save_home):
    """

    :param home: 开源数据集路径
    :param save_home: 保存数据集路径
    :return:
    """
    data = []
    for name in os.listdir(home):
        if not name.endswith('json'):
            continue
        path2 = os.path.join(home, name)
        print(path2)
        with open(path2, 'r', encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                query = sample['QA'][0]['question']
                pos = [sample['positive_doc'][0]['text']]
                negative_doc = sample['negative_doc']
                neg = [negative_doc[idx]['text'] for idx in random.choices(range(len(negative_doc)), k=10)]
                data.append(json.dumps({"query": query, "pos": pos, "neg": neg}, ensure_ascii=False))

    size = int(len(data) * 0.01)
    train = data[size:]
    dev = data[:size]
    print(len(train), len(dev))
    with open(os.path.join(save_home, 'train.jsonl'), 'w', encoding="utf-8") as f:
        f.writelines('\n'.join(train))
    with open(os.path.join(save_home, 'dev.jsonl'), 'w', encoding="utf-8") as f:
        f.writelines('\n'.join(dev))

if __name__ == "__main__":
    args = parse_args()
    get_data(home=args.data_path, save_home=args.save_home)