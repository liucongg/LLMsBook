import argparse
from FlagEmbedding import FlagModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="", help='模型id或local path')
    return parser.parse_args()

def get_similarity(model_path):
    questions = ["孩子为什么容易在手术前感到紧张或焦虑？"]
    paragraph = ["即使是小手术,也会让人紧张不已。孩子们很容易在手术前感到紧张或焦虑,因为他们不太容易理解复杂的医学术语。此外,医生也不愿意向16岁以下的儿童开抗焦虑药物。", "一到周末就睡到中午?专家提醒赖床补觉只会适得其反\n\n经过了一个星期的工作和学习,你是不是一到周末就想睡到自然醒？"]
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                      use_fp16=True)
    embeddings_1 = model.encode(questions)
    embeddings_2 = model.encode(paragraph)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)

if __name__ == '__main__':
    args = parse_args()
    get_similarity(args.model_name_or_path)