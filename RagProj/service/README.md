本项目为书籍《大型语言模型实战指南：应用实践与场景落地》中第9章《基于知识库的大型语言模型问答应用》实战部分代码-基于Streamlit的知识库答案应用搭建。

## 项目简介

本项目是基于Streamlit搭建的知识库答案生成应用。

项目主要结构如下：
- web_service：Streamlit综合服务。
  - web.py：Streamlit问答主函数。
  - split.py：用于针对输入文档进行拆分的方法。

## 服务运行

模型训练需要运行web.py文件进行streamlit 加载

命令如下：
- 模型训练训练
```shell
cd service/web_service

streamlit run  web.py -- --server.port 1111 \
                      -- --embed_model_path 'your embedding model path' \
                      -- --model_path 'your llm model path'


```
## 总结

本项目中的代码包含大量的注释信息，帮助读者更容易的阅读代码、以及了解其原理。读者跑通代码的后，可以根据自己特定的任务，定向修改配置参数或代码，实现自己响应的功能。