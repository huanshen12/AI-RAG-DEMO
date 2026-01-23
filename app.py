# app.py
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_backend import ask_document # 导入刚才写的后端函数

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="智能文档问答助手",
    page_icon="📄",
    layout="wide"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tmp_path" not in st.session_state:
    st.session_state.tmp_path = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# 侧边栏
with st.sidebar:
    st.title("📄 智能文档问答助手")
    
    # 上传文档
    st.header("1. 上传文档")
    uploaded_file = st.file_uploader("请上传一个 PDF 文件", type="pdf")
    
    # 保存上传的文件
    if uploaded_file:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.tmp_path = tmp_file.name
                st.session_state.uploaded_file_name = uploaded_file.name
            st.success(f"✅ 成功上传文件：{uploaded_file.name}")
    
    # API Key 设置
    st.header("2. API Key 设置")
    api_key = st.text_input("请输入 Gitee AI API Key", type="password")
    
    # 参数设置
    st.header("3. 参数设置")
    chunk_size = st.slider("文本分割大小", min_value=200, max_value=1000, value=500, step=50)
    chunk_overlap = st.slider("文本重叠大小", min_value=0, max_value=100, value=50, step=10)
    top_k = st.slider("检索文档数量", min_value=1, max_value=5, value=3, step=1)
    
    # 清除对话历史
    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.success("✅ 对话历史已清除")

# 主聊天界面
st.title("💬 对话界面")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("关于这个文档，你想知道什么？"):
    # 验证必要条件
    if not api_key:
        st.error("请先在侧边栏输入 API Key！")
    elif not st.session_state.tmp_path:
        st.error("请先在侧边栏上传 PDF 文件！")
    else:
        # 添加用户消息到聊天历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 显示 AI 思考中
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 正在思考中...")
            
            try:
                # 构建完整的对话历史，用于上下文
                chat_history = []
                for msg in st.session_state.messages[:-1]:  # 排除当前用户消息
                    if msg["role"] == "user":
                        chat_history.append(f"用户: {msg['content']}")
                    else:
                        chat_history.append(f"AI: {msg['content']}")
                
                # 构建完整的上下文
                context_str = "\n".join(chat_history)
                full_query = f"""
                以下是之前的对话历史：
                {context_str}
                
                请基于之前的对话和文档内容，回答用户的最新问题：
                {prompt}
                """
                
                # 调用后端函数
                answer = ask_document(st.session_state.tmp_path, full_query, api_key)
                
                # 更新消息
                message_placeholder.markdown(answer)
                
                # 添加 AI 回答到聊天历史
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ 出错啦：{str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})