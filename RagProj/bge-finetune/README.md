本项目为书籍《大型语言模型实战指南：应用实践与场景落地》中第9章《基于知识库的大型语言模型问答应用》实战基于知识库的大型语言模型问答实战中BGE微调代码。

## 项目简介

本项目是基于BGE模型的文本表征模型。利用bge-base-zh模型从开源数据中进行长文本表征任务微调，并利用对比学习方法进行数据构造。

项目主要结构如下：
- data：存放数据及数据处理的文件夹。
  - dev.jsonl：验证集数据。
  - train.jsonl：训练数据。
  - load_data.py：用于针对开源数据进行数据处理，生成训练集及验证集数据。
- finetune：模型训练的文件夹。
  - arguments.py：BGE训练中相关的配置信息。
  - data.py：BGE训练时所需要构建的数据格式。
  - modeling.py：BGE模型文件。
  - run.py：BGE训练主函数。
  - trainer.py：BGE训练的trainer方法。
- predict：推理所需的代码文件夹。
  - get_embedding.py：利用已训练的模型进行文本表征并计算相似度

## 数据处理

load_data.py文件，会在data文件夹中生成训练集和测试集文件。

命令如下：

```shell
cd data

python3 load_data.py
```
本项目中的数据来源于开源社区huggingface.co中的Multi-Doc-QA-Chinese，参考文档源数据来自悟道开源200GB数据，其中问题和回答是通过大语言模型（GPT-3.5）自动生成的，并且具有高质量。原始数据集中，每个样本包含一个参考文档、99个无关文档、一个问题和一个基于参考文档的回答。

数据地址为：https://huggingface.co/datasets/yuyijiong/Multi-Doc-QA-Chinese

注意：如果需要修改数据生成路径或名称，load_data.py中相关配置的路径。

## 模型微调

模型训练需要运行run.py文件，会自动生成output_dir文件夹，存放每个save_model_step保存的模型文件。

本项目中采用BGE模型进行微调，模型为huggiFace中的 BAAI/bge-base-zh，路径为：https://huggingface.co/BAAI/bge-base-zh

命令如下：
- 模型训练训练
```shell
cd finetune

python3 run.py --model_name_or_path "your_bge_model_path" \
               --data_path "your_train_data_path" \
               --output_dir ./save_files
               
```

## 模型推理

模型融合执行命令：
```shell
cd predict

python3 get_embedding.py

```
## 总结

本项目中的代码包含大量的注释信息，帮助读者更容易的阅读代码、以及了解其原理。读者跑通代码的后，可以根据自己特定的任务，定向修改配置参数或代码，实现自己响应的功能。