# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: web_demo
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2024/2/4 15:55
"""
    文件说明：
            
"""
import gradio as gr
import argparse
from transformers.generation.utils import GenerationConfig
import json
import torch
from baichuan.tokenization_baichuan import BaichuanTokenizer
from baichuan.modeling_baichuan import BaichuanForCausalLM


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1", help="")
    parser.add_argument("--model_path", type=str, default="output_dir/epoch-1-step-1002-merge/", help="")
    parser.add_argument("--desc_path", type=str, default="data/profiles-zh_desc.json", help="")
    return parser.parse_args()


def predict(input, chatbot, role):
    """
    单条预测函数
    Args:
        input: 用户输入
        chatbot: 机器人组件
        role: 角色

    Returns:

    """
    chatbot.append((input, ""))
    messages = []
    messages.append({"role": "system",
                     "content": "你是{}，你的特征描述是：{}。现在请你回答我的一些问题，以准确展现你的人格特征！你的说话风格要全面模仿被赋予的人格角色！请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要嗦，也不要太过于正式或礼貌。".format(
                         role, desc_data[role])})
    messages.append({"role": "user", "content": input})
    response = model.chat(tokenizer, messages)
    chatbot[-1] = (input, response)
    return chatbot


if __name__ == '__main__':
    # 初始化配置信息
    args = parse_args()
    # 加载Baichuan2的模型和Tokenizer
    tokenizer = BaichuanTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = BaichuanForCausalLM.from_pretrained(args.model_path, device_map="cuda:{}".format(args.device),
                                                torch_dtype=torch.bfloat16)
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(args.model_path)
    # 加载角色描述信息
    with open(args.desc_path, "r", encoding="utf-8") as fh:
        desc_data = json.load(fh)
    # 创建自定义的交互式Web应用和演示
    with gr.Blocks() as demo:
        # 创建一个chatbot机器人组件
        chatbot = gr.Chatbot(label='Role-Playing', elem_classes="control-height")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    # 定义用户输入框
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=13).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    # 定义提交按钮
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                # 定义清楚按钮
                emptyBtn = gr.Button("Clear History")
                # 定义单选角色框、模型输出最大长度以及模型解码Top-p值
                role = gr.Radio(["孙悟空", "李白", "张飞"], value="孙悟空", label="Role", interactive=True)
                max_tgt_len = gr.Slider(0, 2048, value=512, step=1.0, label="Maximum tgt length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
        model.generation_config.top_p = float(top_p.value)
        model.generation_config.max_new_tokens = int(max_tgt_len.value)
        # 当点击提交按钮后，调用单条预测函数，并将输入框清空
        submitBtn.click(predict, [user_input, chatbot, role], [chatbot], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        # 点击清空按钮后，重新初始化chatbot机器人组件
        emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)
    # 运行demo，设置IP以及对应端口号
    demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=9090)
