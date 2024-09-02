本项目为书籍《大型语言模型实战指南：应用实践与场景落地》中第5章《大型语言模型SQL任务实战》实战部分代码-Text2SQL任务实战。

## 项目简介

本项目是基于DeepSeek Coder进行SQL生成任务进行的微调方法介绍。利用DeepSeek Coder模型从开源数据中进行数据构造，并进行模型微调。

项目主要结构如下：
- data：存放数据及数据处理的文件夹。
  - dev.jsonl：验证集数据。
  - train.jsonl：训练数据。
  - dusql_process.py：用于针对开源数据进行数据处理，生成训练集及验证集数据。
- finetune：模型训练的文件夹。
  - train_deepseek.py：使用DeepSeek Coder模型训练的函数。
- predict：推理所需的代码文件夹。
  - predict.py：利用已训练的模型进行模型生成的方法。

## 数据处理

数据预处理需要运行dusql_process.py文件，会在data文件夹中生成训练集和测试集文件。

命令如下：

```shell
cd data

python3 dusql_process.py
```
本次微调主要针对[dusql数据](https://aistudio.baidu.com/competition/detail/47/0/task-definition) 进行应用，并且由于当前dusql数据中，表格信息以中文为主，因此本次我们还将采用翻译模型对数据中的表格字段信息进行翻译，翻译器可以使用开源中-英翻译模型。


本项目中采用huggiFace中的 [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)

注意：如果需要修改数据生成路径或名称，请修改data_helper.py中相关配置的路径。

## 模型微调

模型训练需要运行train.py文件，会自动生成output_dir文件夹，存放每个save_model_step保存的模型文件。

本项目中采用deepseek 模型进行微调，模型为huggiFace中的 [deepseek-ai/deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)

命令如下：
- 模型训练训练
```shell
cd finetune
python3 train_deepseek.py --model_name_or_path "your_deepseek_model_path" \
                          --data_path "your_train_data_path" \
                          --output_dir ./save_files
```

## 模型推理

模型融合执行命令：
```shell
cd predict
python3 predict.py --model_path "your_model_path"
```
## 总结

本项目中的代码包含大量的注释信息，帮助读者更容易的阅读代码、以及了解其原理。读者跑通代码的后，可以根据自己特定的任务，定向修改配置参数或代码，实现自己响应的功能。