本项目为书籍《大型语言模型实战指南：应用实践与场景落地》中第2章《大型语言模型常用微调方法》中所提及的tokenizer训练方法

## 项目简介

本项目分别介绍了bpe、unigram、wordpiece、sentencepiece的分词器训练方法

项目主要结构如下：
- dataset：存放数据的文件夹。
- bpe_module：bpe训练的文件夹。
  - train_bpe.py：训练bpe分词器代码。
- unigram_module：unigram训练的文件夹。
  - train_unigram.py：训练unigram分词器代码。
- wordpiece_module：wordpiece训练的文件夹。
  - train_wordpiece.py：训练wordpiece分词器代码。
- sentencepiece_module：sentencepiece训练的文件夹。
  - train_sentencepiece.py：训练sentencepiece分词器代码。

## bpe训练

模型训练需要运行train_bpe.py文件。

命令如下：
- 模型训练训练
```shell
cd bpe_module
python3 train_bpe.py --file_path ../dataset  \
                     --save_path ./models  \
                     --vocab_size 10000
```

## unigram训练

模型训练需要运行train_unigram.py文件。

命令如下：
- 模型训练训练
```shell
cd unigram_module
python3 train_unigram.py --file_path ../dataset  \
                         --save_path ./models  \
                         --vocab_size 10000
```

## wordpiece训练

模型训练需要运行train_wordpiece.py文件。

命令如下：
- 模型训练训练
```shell
cd wordpiece_module
python3 train_wordpiece.py --file_path ../dataset  \
                           --save_path ./models  \
                           --vocab_size 10000
```

## wordpiece训练

模型训练需要运行train_wordpiece.py文件。

命令如下：
- 模型训练训练
```shell
cd sentencepiece_module
python3 train_sentencepiece.py --file_path ../dataset/lines.txt  \
                               --save_path ./models/spm_model  \
                               --vocab_size 10000
```


## 总结

本项目中的代码包含大量的注释信息，帮助读者更容易的阅读代码、以及了解其原理。读者跑通代码的后，可以根据自己特定的任务，定向修改配置参数或代码，实现自己响应的功能。