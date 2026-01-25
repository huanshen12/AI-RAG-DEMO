# 📚 RAG-Knowledge-Engine (基于微服务架构的文档问答引擎)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-orange)](https://www.langchain.com/)

## 📖 项目简介 (Introduction)

本项目是一个轻量级但功能完备的 **RAG (Retrieval-Augmented Generation)** 知识库问答引擎。

不同于常见的单脚本 Demo，本项目采用了**前后端分离**的微服务架构设计。后端基于 **FastAPI** 提供标准化的 RESTful 接口，前端使用 **Streamlit** 构建交互界面。系统内置了**二级向量缓存机制**与**流式响应管道**，旨在解决大模型应用中常见的“响应慢”与“等待久”的痛点，提供企业级的交互体验。

---

## ✨ 核心特性 (Key Features)

### 1. 🏗 微服务架构 (Microservices Architecture)
* **解耦设计**：将计算密集型的 RAG 逻辑封装在 FastAPI 后端，前端只负责渲染。这种设计支持独立部署和扩展，方便未来对接微信小程序、Web 或其他客户端。
* **标准化接口**：提供 `/chat` (标准) 和 `/chat/stream` (流式) 两套接口，满足不同场景需求。

### 2. ⚡️ 智能全局缓存 (Smart In-Memory Cache)
* **性能瓶颈突破**：针对 RAG 系统中耗时最长的 "Embedding 向量化" 环节，实现了基于内存的全局缓存池 (`VECTOR_STORE_CACHE`)。
* **秒级响应**：对于同一文档的后续提问，系统自动命中缓存，跳过加载与向量化步骤，将响应延迟从 **10s+ 降低至 <200ms**。

### 3. 🌊 全链路流式响应 (Full-Stack Streaming)
* **极致体验**：打通了 `LLM -> Backend (Yield) -> Frontend` 的完整流式传输链路。
* **实时反馈**：实现类似 ChatGPT 的“打字机”生成效果，彻底告别由长文本生成导致的界面假死。

### 4. 🔌 灵活的组件适配
* **自定义封装**：重写了 LangChain 的 `Embeddings` 接口 (`embeddings.py`)，实现了对 **Gitee AI / OpenAI** 格式接口的无缝适配。
* **模型无关性**：底层逻辑不绑定特定模型，可轻松切换 DeepSeek、Qwen 或本地部署的 Llama 模型。

---

## 🛠️ 技术栈 (Tech Stack)

| 模块 | 技术选型 | 作用 |
| :--- | :--- | :--- |
| **Backend** | **FastAPI** | 高性能异步 Web 框架，负责业务逻辑与接口暴露 |
| **Frontend** | **Streamlit** | 数据可视化与交互界面 |
| **Orchestration** | **LangChain** | RAG 流程编排、Prompt 模板管理 |
| **Vector DB** | **FAISS** | 本地向量索引与检索 |
| **Model Provider** | **Gitee AI API** | 提供 DeepSeek-V3 推理与 Qwen 向量化能力 |
| **Validation** | **Pydantic** | 数据模型定义与运行时校验 |

---
## 📂 项目结构
```Plaintext
.
├── api.py                 # [Entry] FastAPI 后端入口，定义 RESTful 接口
├── app.py                 # [Entry] Streamlit 前端入口，处理 UI 交互
├── rag_backend.py         # [Core] RAG 核心引擎 (加载/切分/缓存/检索)
├── embeddings.py          # [Driver] 自定义向量模型驱动器
├── requirements.txt       # 项目依赖清单
└── .env                   # 敏感配置信息
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
在项目根目录创建 .env 文件，填入你在 Gitee AI 申请的 API Key 以及你申请的要使用的大模型API Key和URL

```Plaintext
GITEE_AI_API_KEY=你的_Gitee_API_Key
```
4. 运行应用
```Bash

uvicorn api:app --reload
```
```Bash

streamlit run app.py
```
## 📝 开发日志
```text
[x] V1.0: 实现基础 RAG 流程与 Streamlit 界面

[x] V1.1: 重构为前后端分离架构，引入 FastAPI

[x] V1.2: 实现流式输出 (Streaming) 与 全局向量缓存 (Caching)

[ ] V2.0: 引入 Redis 替代内存缓存，实现持久化存储

[ ] V2.1: 支持多文件上传与知识库管理


```
**本项目仅供学习使用，API 额度请自行管理。**
