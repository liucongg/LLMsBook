# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: train
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2024/1/31 21:15
"""
    文件说明：
            
"""
import argparse
import json
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from transformers import BitsAndBytesConfig
from utils import print_trainable_parameters, print_rank_0, to_device, set_random_seed, save_model, DataCollator, \
    find_all_linear_names, evaluation
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from baichuan.modeling_baichuan import BaichuanForCausalLM
from baichuan.tokenization_baichuan import BaichuanTokenizer
from baichuan.configuration_baichuan import BaichuanConfig
from utils import Baichuan2PromptDataSet
from transformers import GenerationConfig
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型配置
    parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    # 数据配置
    parser.add_argument("--train_path", default="", type=str, help="")
    parser.add_argument("--test_path", default="", type=str, help="")
    parser.add_argument("--max_len", type=int, default=1024, help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--is_skip", action='store_true', help="")
    # 训练配置
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    # DeepSpeed配置
    parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
    # QLoRA配置
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=30, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def train():
    # 设置模型训练参数
    args = parse_args()
    # 判断是多卡训练还是单卡训练
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()
    # 设置tensorboard，记录训练过程中的loss以及ppl
    if args.global_rank <= 0:
        tb_write = SummaryWriter()
    # 设置随机种子，方便模型复现
    set_random_seed(args.seed)
    torch.distributed.barrier()
    # 加载百川模型分词器
    tokenizer = BaichuanTokenizer.from_pretrained(args.model_name_or_path)
    # 加载百川模型
    device_map = {'': int(os.environ.get('LOCAL_RANK', '0'))}
    model_config = BaichuanConfig.from_pretrained(args.model_name_or_path)
    model = BaichuanForCausalLM.from_pretrained(args.model_name_or_path,
                                                quantization_config=BitsAndBytesConfig(
                                                    load_in_4bit=True,
                                                    bnb_4bit_compute_dtype=model_config.torch_dtype,
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    llm_int8_threshold=6.0,
                                                    llm_int8_has_fp16_weight=False,
                                                ),
                                                torch_dtype=model_config.torch_dtype,
                                                device_map=device_map)
    # model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
    model = prepare_model_for_kbit_training(model)
    # 找到模型中所有的全连接层
    lora_module_name = find_all_linear_names(model)
    # 设置Lora配置，并生成外挂可训练参数
    config = LoraConfig(r=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        target_modules=lora_module_name,
                        lora_dropout=args.lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )
    model = get_peft_model(model, config)
    model.config.torch_dtype = torch.float32
    # 打印可训练参数
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print_rank_0(name, 0)
    print_trainable_parameters(model)
    print(model.generation_config)

    # 加载模型训练所需要的数据，如果是多卡训练需要分布式加载数据
    train_dataset = Baichuan2PromptDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len,
                                           model.generation_config, args.is_skip)
    test_dataset = Baichuan2PromptDataSet(args.test_path, tokenizer, args.max_len, args.max_src_len,
                                          model.generation_config, args.is_skip)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

    data_collator = DataCollator(tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, sampler=test_sampler,
                                 batch_size=args.per_device_train_batch_size)
    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)

    # 加载DeepSpeed配置文件，并进行修改
    with open(args.ds_file, "r", encoding="utf-8") as fh:
        ds_config = json.load(fh)
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    # load optimizer
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
    ds_config["optimizer"]["params"]["betas"] = (0.9, 0.95)
    ds_config["optimizer"]["params"]["eps"] = 1e-8
    ds_config["optimizer"]["params"]["weight_decay"] = 0.1
    num_training_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print_rank_0("num_training_steps = {}".format(num_training_steps), args.global_rank)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    print_rank_0("num_warmup_steps = {}".format(num_warmup_steps), args.global_rank)
    ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_min_lr"] = args.learning_rate * 0.1

    # 设置模型gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # DeepSeed对模型进行初始化
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, args=args, config=ds_config,
                                                             dist_init_required=True)
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # 模型开始训练
    for epoch in range(args.num_train_epochs):
        print_rank_0("Beginning of Epoch {}/{}, Total Micro Batches {}".format(epoch + 1, args.num_train_epochs,
                                                                               len(train_dataloader)), args.global_rank)
        model.train()
        # 遍历所有数据
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            batch = to_device(batch, device)
            # 获取训练结果
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # 损失进行回传
            model.backward(loss)
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.step()
            # 当训练步数整除累积步数时，记录训练损失值和模型保存
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # 损失值记录
                if global_step % args.show_loss_step == 0:
                    if args.global_rank <= 0:
                        tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / (
                                args.show_loss_step * args.gradient_accumulation_steps), global_step)
                        logging_loss = tr_loss
                # 模型保存并验证测试集的PPL值
                if args.save_model_step is not None and global_step % args.save_model_step == 0:
                    ppl = evaluation(model, test_dataloader, device)
                    if args.global_rank <= 0:
                        tb_write.add_scalar("ppl", ppl, global_step)
                        print_rank_0("save_model_step-{}: ppl-{}".format(global_step, ppl), args.global_rank)
                    if args.global_rank <= 0:
                        save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
                    model.train()
        # 每个Epoch对模型进行一次测试，记录测试集的损失
        ppl = evaluation(model, test_dataloader, device)
        if args.global_rank <= 0:
            tb_write.add_scalar("ppl", ppl, global_step)
            print_rank_0("save_model_step-{}: ppl-{}".format(global_step, ppl), args.global_rank)
        if args.global_rank <= 0:
            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")


if __name__ == "__main__":
    train()
