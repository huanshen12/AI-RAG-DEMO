# 🤖 AI-RAG-Demo (基于 Gitee AI 的文档问答助手)

## 📖 项目介绍
这是一个基于 **RAG (检索增强生成)** 技术构建的本地知识库问答应用。
用户上传 PDF 文档后，系统会对文档进行切分、向量化存储。当用户提问时，系统会从文档中检索相关片段，并结合 **DeepSeek-V3** 大模型生成精准回答。

本项目是一个纯 Python 实现的 Demo，重点展示了如何自定义 LangChain 的 Embeddings 接口以适配 Gitee AI 平台。

## 🛠️ 技术栈 (Tech Stack)

* **大模型 (LLM)**: DeepSeek-V3 (通过 Gitee AI Serverless API 调用)
* **向量模型 (Embeddings)**: Qwen3-Embedding-8B (自定义封装 Gitee AI 接口)
* **向量数据库**: FAISS (本地内存级向量存储)
* **开发框架**: LangChain + LangChain-Community
* **Web 界面**: Streamlit
* **文档处理**: PyPDFLoader (加载) + RecursiveCharacterTextSplitter (切分)

## 📂 项目结构
```text
.
├── app.py                 # Streamlit 前端入口
├── rag_backend.py         # RAG 核心逻辑 (加载、切分、检索、生成)
├── embeddings.py          # 自定义 Gitee AI 向量模型封装类
├── requirements.txt       # 项目依赖包
├── .env                   # 环境变量 (存放 API Key)
└── README.md              # 项目说明文档
```
## 🚀 快速启动

1. 克隆项目
```Bash

git clone [https://github.com/huanshen12/AI-RAG-DEMO.git](https://github.com/huanshen12/AI-RAG-DEMO.git)
cd AI-RAG-DEMO
```
2. 安装依赖
```Bash

pip install -r requirements.txt
```
3. 配置环境变量
在项目根目录创建 .env 文件，填入你在 Gitee AI 申请的 API Key：

```Plaintext
GITEE_AI_API_KEY=你的_Gitee_API_Key
```
4. 运行应用
```Bash

streamlit run app.py
```
## 📝 开发日志
```text
[x] 完成 Gitee AI Embeddings 的自定义封装 (embeddings.py)

[x] 集成 FAISS 实现本地向量检索

[x] 接入 DeepSeek-V3 实现问答生成

[x] 计划：优化 UI 界面，增加历史对话记录

[ ] 计划：将核心逻辑代码重构到 core/ 文件夹中
```
**本项目仅供学习使用，API 额度请自行管理。**
