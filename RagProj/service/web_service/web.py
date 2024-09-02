import json
import time
from tempfile import NamedTemporaryFile
import os

import streamlit as st
import argparse
from flag_models import FlagModel
from transformers import AutoTokenizer, AutoModel

from streamlit_chat import message
from split import load_file
import numpy as np
import faiss

# langchain embedding
st.set_page_config(page_title="基于知识库的大型语言模型问答")
st.title("基于知识库的大型语言模型问答")


def get_args():
    parser = argparse.ArgumentParser(description='data helper')
    parser.add_argument('--embed_model_path', type=str, help='向量表征模型路径')
    parser.add_argument('--model_path', type=str, default="./", help="大模型路径")

    return parser.parse_args()


class EmbeddingService:
    def __init__(self, args):
        self.embed_model_path = args.embed_model_path
        self.embed_model = self.load_embedding_model(self.embed_model_path)

    def load_embedding_model(self, model_path):
        embed_model = FlagModel(model_path,
                                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                use_fp16=True)
        return embed_model

    def get_embedding(self, doc_info):
        doc_vectors = self.embed_model.encode(doc_info)
        doc_vectors = np.stack(doc_vectors).astype('float32')
        dimension = 512
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_vectors)
        return index

    def search_doc(self, query, doc_info, index: faiss.IndexFlatL2, k: int):
        query_vector = self.embed_model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        distances, indexes = index.search(query_vector, k)
        found_docs = []
        for i, (distance, index) in enumerate(zip(distances[0], indexes[0])):
            print(f"Result {i + 1}, Distance: {distance}")
            found_docs.append(doc_info[i])
        return found_docs


class GLMService:
    def __init__(self, args):
        self.args = args
        self.glm_model, self.tokenizer = self.init_model(self.args.model_path)

    def init_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        model = model.eval()
        return model, tokenizer

    def get_result(self, doc, question):
        input_ = f"你现在是一个可以根据文档内容进行问答的机器人，以下是用于参考的文档内容：\n\n{doc}\n问题为：{question}\n答："
        response, history = self.glm_model.chat(self.tokenizer, input_, history=[])
        return response


def clear_chat_history():
    del st.session_state.messages
    st.session_state.history = []


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是AI助手，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


# 初始化变量
if 'history' not in st.session_state:
    st.session_state.history = []

# 初始化 session_state
if "enter_pressed" not in st.session_state:
    st.session_state.enter_pressed = False


def generate_response(user_input):
    info = "根据原文“孩子们很容易在手术前感到紧张或焦虑,因为他们不太容易理解复杂的医学术语。此外,医生也不愿意向16岁以下的儿童开抗焦虑药物。”可知，孩子们容易在手术前感到紧张或焦虑是因为他们不太容易理解复杂的医学术语，而且医生也不愿意向16岁以下的儿童开抗焦虑药物。"
    return info


def main():
    # 初始化ChatGLM3-6模型
    # st.markdown("开始初始化，请稍后...")
    st.session_state.glm_service = GLMService(args)
    print("llm is loaded")

    # 初始化BGE向量模型
    st.session_state.embedding_service = EmbeddingService(args)
    print("embedding_service is loaded")

    lines = ["1", "2", "3"]
    st.session_state.embedding_service.get_embedding(lines)
    st.markdown("初始化完成，请上传文件")
    # 获取知识库文件
    uploaded_file = st.file_uploader("请上传文件")
    temp_file = NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    temp_file.write(uploaded_file.getvalue())
    # 构造包含扩展名的临时文件路径
    file_path = temp_file.name
    with st.spinner('Reading file...'):
        texts = load_file(file_path)
    st.success('Finished reading file.')
    temp_file.close()

    # 初始化文档，对文档内容进行向量化
    st.session_state.index = st.session_state.embedding_service.get_embedding(texts)

    # 获取用户问题
    st.markdown("#### 请在下列文框中输入您的问题：")
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    user_input = st.text_input("请输入您的问题:", key='input')

    if user_input:
        # 获取用户问题，并得到向量表征，利用已加载的Faiss获取K个相关doc
        found_docs = st.session_state.embedding_service.search_doc(user_input, texts, st.session_state.index, k=3)
        st.markdown("#### 检索到以下内容：")
        for one in found_docs:
            st.markdown(f"{one}")

        found_doc = "\n".join(found_docs)

        # 生成答案并返回结果展示
        output = st.session_state.glm_service.get_result(found_doc, user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i],
                    is_user=True, key=str(i) + '_user')
            st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    args = get_args()
    main()
