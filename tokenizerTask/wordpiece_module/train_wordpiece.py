import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_wordpiece_model(files_dir=None, save_dir=None, vocab_size=10000):
    """
    训练BPE模型并保存

    参数:
        files (list): 训练数据文件列表
        save_path (str): 保存模型的路径
    """
    if files_dir is None:
        files = [f"../dataset/lines.txt"]
        print(os.path.exists(files[0]))
    else:
        files = [os.path.join(files_dir, f_name) for f_name in os.listdir(files_dir)]

    if save_dir is None:
        save_dir = "./models/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "tokenizer.json")

    # Step 1: 创建一个空白的BPE tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    # Step 2: 实例化BPE tokenizer的训练器
    trainer = WordPieceTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=1,
        show_progress=True,
        vocab_size=vocab_size
    )

    # Step 3: 定义预分词规则（这里使用空格预切分）
    tokenizer.pre_tokenizer = Whitespace()

    # Step 4: 加载数据集，训练tokenizer
    tokenizer.train(files, trainer)

    # Step 5: 保存tokenizer
    tokenizer.save(save_path)


def wordpiece_tokenizer(save_path=None):
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
    # output.tokens: ['我', '##爱', '##北京', '##天', '##安', '##门', '，', '天', '##安', '##门', '##上', '##太阳', '##升', '。']
    # output.ids: [1315, 4110, 8829, 3798, 4077, 4387, 3514, 878, 4077, 4387, 3551, 7568, 3888, 181]

    # 使用tokenizer对句子进行分词
    sentence2 = "A large language model (LLM) is a language model consisting of a neural network with many parameters."
    output2 = tokenizer.encode(sentence2)

    # 打印分词结果
    print("sentence: ", sentence2)
    print("output.tokens: ", output2.tokens)
    print("output.ids: ", output2.ids)

    # 打印结果：
    # sentence:  A large language model (LLM) is a language model consisting of a neural network with many parameters.
    # output.tokens: ['A', 'l', '##ar', '##ge', 'lang', '##u', '##age', 'mod', '##el', '(', 'L', '##L', '##M', ')', 'is', 'a', 'lang', '##u', '##age', 'mod',
    #                 '##el', 'cons', '##ist', '##ing', 'of', 'a', 'ne', '##ur', '##al', 'ne', '##t', '##w', '##or', '##k', 'with', 'ma', '##n', '##y', 'p',
    #                 '##ara', '##m', '##eter', '##s', '.']
    # output.ids: [37, 80, 6962, 7257, 9885, 3782, 8043, 9287, 6998, 12, 48, 4302, 4453, 13, 7141, 69, 9885, 3782, 8043, 9287, 6998, 9623, 7290, 6975, 7074, 69,
    #              7718, 7154, 7000, 7718, 3960, 3867, 6948, 4427, 7625, 8226, 3757, 3760, 84, 9111, 3959, 9229, 3940, 18]

def parse_args():
    parser = argparse.ArgumentParser(description='train bpe tokenizer')
    parser.add_argument('--file_path', type=str, default=None, help='用于训练tokenizer的数据的文件路径')
    parser.add_argument('--save_path', type=str, default=None, help='保存tokenizer模型的路径')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词表规模')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 训练BPE模型
    train_wordpiece_model(files_dir=args.file_path, save_dir=args.save_path, vocab_size=args.vocab_size)

    # 使用BPE tokenizer进行分词
    wordpiece_tokenizer()
