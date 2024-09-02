import json
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='data helper')
    parser.add_argument('--data_path', type=str, default="./Belle_open_source_0.5M.json", help='Belle数据地址')
    parser.add_argument('--train_data_path', type=str, default="./train.jsonl", help='训练数据路径')
    parser.add_argument('--eval_data_path', type=str, default="./dev.jsonl", help='验证数据路径')
    parser.add_argument('--ratio', type=float, default=0.95, help='训练集占比')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def split_data(args):
    with open(args.data_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        train_size = int(len(lines)*args.ratio)
        train_data = lines[:train_size]
        eval_data = lines[train_size:]
    with open(args.train_data_path, 'w', encoding="utf-8") as f:
        f.writelines(train_data)
    with open(args.eval_data_path, 'w', encoding="utf-8") as f:
        f.writelines(eval_data)
    print(f"Data saved\n train_size: {train_size}\n eval_size: {len(lines)-train_size}")
        
        
if __name__ == "__main__":
    args = parse_args()
    split_data(args=args)