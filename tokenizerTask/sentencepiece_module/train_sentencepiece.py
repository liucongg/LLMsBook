import argparse

import sentencepiece as spm
import os

def train_sentencepiece(corpus_path, save_dir, vocab_size):
    """
    sentencepiece训练

    Args:
        corpus_path (str): 训练文本路径
        save_dir (str): 模型保存路径
        vocab_size (int): 词表大小

    Returns:
        spm.SentencePieceProcessor: 训练得到的SentencePiece分词器
    """
    # Train the SentencePiece tokenizer
    if corpus_path is None:
        corpus_path = f"../dataset/lines.txt"

    if save_dir is None:
        save_dir = "./models/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_prefix = os.path.join(save_dir, "spm_model")
    print(f"model_prefix: {model_prefix}")
    spm.SentencePieceTrainer.train(input=corpus_path, model_prefix=model_prefix, vocab_size=vocab_size,
                                   model_type="bpe")

    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

def do_predict(tokenizer):
    # 测试用例
    sentence = "我爱北京天安门，天安门上太阳升。"
    encoded_sentence = tokenizer.encode_as_pieces(sentence)
    print("Encoded sentence 1:", encoded_sentence)
    # 打印结果
    # Encoded sentence 1: ['▁我', '爱', '北京', '天', '安', '门', ',', '天', '安', '门', '上', '太阳', '升', '。']

    sentence2 = "A large language model (LLM) is a language model consisting of a neural network with many parameters. "
    encoded_sentence2 = tokenizer.encode_as_pieces(sentence2)
    print("Encoded sentence 2:", encoded_sentence2)
    # 打印结果
    # Encoded sentence 2: ['▁A', '▁lar', 'ge', '▁language', '▁model', '▁(', 'L', 'L', 'M', ')', '▁is', '▁a', '▁language', '▁model', '▁consist', 'ing', '▁of', '▁a', '▁ne', 'ur', 'al', '▁n', 'et', 'w', 'ork', '▁with', '▁many', '▁par', 'am', 'et', 'ers', '.']



def parse_args():
    parser = argparse.ArgumentParser(description='train bpe tokenizer')
    parser.add_argument('--file_path', type=str, default=None, help='用于训练tokenizer的数据的文件路径')
    parser.add_argument('--save_path', type=str, default=None, help='保存tokenizer模型的路径')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词表规模')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 训练 SentencePiece 分词器
    tokenizer = train_sentencepiece(corpus_path=args.file_path, save_dir=args.save_path, vocab_size=args.vocab_size)

    do_predict(tokenizer)
