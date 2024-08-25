# -*- coding:utf-8 -*-
# @project: PyCharm
# @filename: test.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2024/8/25 16:55
"""
    
"""
import os

os.environ["OPENAI_API_KEY"] = "******"

os.environ["SERPAPI_API_KEY"] = "******"

from langchain.agents import Tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful focr when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


from langchain.chat_models import ChatOpenAI
from agent import AutoGPT

agent = AutoGPT.from_llm_and_tools(
    ai_name="撰写天气报告的BOT",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(),
)
# Set verbose to be true
agent.chain.verbose = True

agent.run(["写一个今天北京的天气报告"])

