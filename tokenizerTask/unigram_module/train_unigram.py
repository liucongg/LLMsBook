import argparse
import os
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_uigram_model(files_dir=None, save_dir=None, vocab_size=10000):
    """
    训练BPE模型并保存

    参数:
        files (list): 训练数据文件列表
        save_path (str): 保存模型的路径
    """
    if files_dir is None:
        files = [f"../dataset/lines.txt"]
    else:
        files = [os.path.join(files_dir, f_name) for f_name in os.listdir(files_dir)]

    if save_dir is None:
        save_dir = "./models/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "tokenizer.json")

    # Step 1: 创建一个空白的BPE tokenizer
    tokenizer = Tokenizer(Unigram())

    # Step 2: 实例化BPE tokenizer的训练器
    trainer = UnigramTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        show_progress=True,
        vocab_size=vocab_size,
        unk_token="[UNK]"
    )

    # Step 3: 定义预分词规则（这里使用空格预切分）
    tokenizer.pre_tokenizer = Whitespace()

    # Step 4: 加载数据集，训练tokenizer
    tokenizer.train(files, trainer)

    # Step 5: 保存tokenizer
    tokenizer.save(save_path)


def unigram_tokenizer(save_path=None):
    """
    使用训练好的BPE tokenizer进行分词

    参数:
        save_path (str): 训练好的BPE tokenizer模型的保存路径
    """
    if save_path is None:
        save_path = "./models/tokenizer.json"

    # 加载tokenizer
    tokenizer = Tokenizer.from_file(save_path)

    # 使用tokenizer对句子进行分词
    sentence = "我爱北京天安门，天安门上太阳升。"
    output = tokenizer.encode(sentence)

    # 打印分词结果
    print("sentence: ", sentence)
    print("output.tokens: ", output.tokens)
    print("output.ids: ", output.ids)

    # 打印结果：
    # sentence: 我爱北京天安门，天安门上太阳升。
    # output.tokens: ['我', '爱', '北京', '天', '安', '门', '，', '天', '安', '门', '上', '太阳', '升', '。']
    # output.ids: [27, 275, 930, 245, 1406, 778, 5, 245, 1406, 778, 40, 1172, 3480, 6]

    # 使用tokenizer对句子进行分词
    sentence2 = "A large language model (LLM) is a language model consisting of a neural network with many parameters."
    output2 = tokenizer.encode(sentence2)

    # 打印分词结果
    print("sentence: ", sentence2)
    print("output.tokens: ", output2.tokens)
    print("output.ids: ", output2.ids)

    # 打印结果：
    # sentence:  A large language model (LLM) is a language model consisting of a neural network with many parameters.
    # output.tokens: ['A', 'large', 'language', 'model', '(', 'L', 'L', 'M', ')', 'is', 'a', 'language', 'model', 'consist', 'ing', 'of', 'a', 'ne', 'ur', 'al',
    #                 'n', 'et', 'work', 'with', 'man', 'y', 'par', 'am', 'et', 'ers', '.']
    # output.ids: [61, 5668, 3664, 3520, 57, 582, 582, 403, 55, 96, 34, 3664, 3520, 5667, 81, 145, 34, 1605, 574, 203, 80, 668, 2229, 649, 1774, 58, 3442, 530,
    #              668, 2294, 11]

def parse_args():
    parser = argparse.ArgumentParser(description='train unigram tokenizer')
    parser.add_argument('--file_path', type=str, default=None, help='用于训练tokenizer的数据的文件夹路径')
    parser.add_argument('--save_path', type=str, default=None, help='保存tokenizer模型的路径')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词表规模')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 训练BPE模型
    train_uigram_model(files_dir=args.file_path, save_dir=args.save_path, vocab_size=args.vocab_size)

    # 使用BPE tokenizer进行分词
    unigram_tokenizer()
