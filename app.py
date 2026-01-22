# app.py
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_backend import ask_document # 导入刚才写的后端函数

# 加载环境变量
load_dotenv()

st.title("📄 智能文档问答助手 (RAG Demo)")

# 1. 左侧侧边栏：上传文件和输入 API Key
with st.sidebar:
    st.header("1. 上传文档")
    uploaded_file = st.file_uploader("请上传一个 PDF 文件", type="pdf")
    
    st.header("2. API Key 设置")
    api_key = st.text_input("请输入 Gitee AI API Key", type="password")


# 2. 主界面：聊天窗口
st.header("3. 提问")
query = st.text_input("关于这个文档，你想知道什么？")

if st.button("开始回答"):
    if not api_key:
        st.error("请先输入 API Key！")
    elif not uploaded_file:
        st.error("请先上传 PDF 文件！")
    elif not query:
        st.error("请输入问题！")
    else:
        with st.spinner("正在阅读文档并思考中..."):
            # 为了给 LangChain 读取，我们需要把上传的文件存成临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # 调用后端函数，传递 API Key
                answer = ask_document(tmp_path, query, api_key)
                st.success("回答完成！")
                st.markdown(f"### 🤖 AI 回复：\n{answer}")
            except Exception as e:
                st.error(f"出错啦：{str(e)}")