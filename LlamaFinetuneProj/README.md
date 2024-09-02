本项目为书籍《大型语言模型实战指南：应用实践与场景落地》中第2章《大型语言模型常用微调方法》实战部分代码-基于PEFT的Llama模型微调实战。

## 项目简介

本项目是基于Llama-2进行PEFT进行的微调方法介绍。利用Llama2-Chinese-7b模型从开源数据中进行数据构造，并进行模型微调。本项目从数据预处理、模型微调和模型推理几个部分入手，手把手地带领大家一起完成Llama-2 PEFT微调任务。

项目主要结构如下：
- data：存放数据及数据处理的文件夹。
  - dev.jsonl：验证集数据。
  - train.jsonl：训练数据。
  - load_data.py：用于针对开源数据进行数据处理，生成训练集及验证集数据。
- finetune：模型训练的文件夹。
  - train_lora_llama.py：使用LoRA进行Llama-2训练的函数。
- predict：推理所需的代码文件夹。
  - predict.py：利用已训练的模型进行模型生成的方法。

## 数据处理

数据预处理需要运行data_helper.py文件，会在data文件夹中生成训练集和测试集文件。

命令如下：

```shell
cd data

python3 data_helper.py --data_path ./Belle_open_source_0.5M.json 

```

注意：如果需要修改数据生成路径或名称，请修改data_helper.py中相关配置的路径。
## 模型微调

模型训练需要运行train.py文件，会自动生成output_dir文件夹，存放每个save_model_step保存的模型文件。

命令如下：
- 模型训练训练
```shell
cd finetune
python3 train_lora_llama.py --train_args_json ./llama2-7B_LoRA.json  \
                            --train_data_path ../data/train.jsonl  \
                            --eval_data_path ../data/dev.jsonl  \
                            --model_name_or_path Llama2-Chinese-7b/  \
                            --seed 42  \
                            --max_input_length 1024  \
                            --max_output_length 1024  \
                            --lora_rank 4  \
                            --lora_dim 8
```

## 模型推理

模型融合执行命令：
```shell
cd predict
python3 predict.py --model_path "your_model_path"
```
## 总结

本项目中的代码包含大量的注释信息，帮助读者更容易的阅读代码、以及了解其原理。读者跑通代码的后，可以根据自己特定的任务，定向修改配置参数或代码，实现自己响应的功能。