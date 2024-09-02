import json
import random
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='data helper')
    parser.add_argument('--data_path', type=str, help='Multi-Doc-QA-Chinese/raw的地址')
    parser.add_argument('--save_home', type=str, default="./", help='训练及验证数据保存路径')
    parser.add_argument('--prompt', type=str, default="你现在扮演一名RAG专家，我将给你一些参考线索，请根据线索回答问题，以下为参考线索：{all_doc_lines} \n 问题：{question} \n 答案为：", help='验证数据路径')
    parser.add_argument('--ratio', type=float, default=0.99, help='训练集占比')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def get_data(home, save_home):
    """
    数据处理
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
                answer = sample['QA'][0]['answer']
                doc = sample['positive_doc'][0]['text']
                one = {
                    "instruction": f"你现在是一个可以根据文档内容进行问答的机器人，以下是用于参考的文档内容：\n\n{doc}\n问题为：{query}\n答：",
                    "output": answer
                }
                data.append(json.dumps(one, ensure_ascii=False))

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