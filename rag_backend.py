# rag_backend.py
import os
import warnings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# --- 正确的导入方式 ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import GiteeAIEmbeddings  # 使用 Gitee AI Embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# 导入必要的模块
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

print("✅ 后端模块已加载 (已启用全局内存缓存)")

# ==========================================
# 🚀 全局缓存字典
# ==========================================
# 这是一个存放在内存里的字典，用来保存处理好的向量库
# Key (键): 文件路径 (例如 "temp/doc.pdf")
# Value (值): 处理好的 FAISS 向量库对象
VECTOR_STORE_CACHE = {}


def get_vectorstore(file_path, api_key):
    """
    核心助手函数：获取向量存储实例（带缓存机制）
    
    逻辑：
    1. 先看缓存里有没有。
    2. 有的话，直接拿来用（秒回！）。
    3. 没有的话，才去辛苦加载、切分、向量化，然后存进缓存供下次用。
    """
    global VECTOR_STORE_CACHE
    
    # --- 1. 检查缓存 ---
    if file_path in VECTOR_STORE_CACHE:
        print(f"⚡ [缓存命中] 发现已处理过的文档: {file_path}")
        print("   -> 跳过加载、切分、向量化，直接复用！")
        return VECTOR_STORE_CACHE[file_path]
    
    # --- 2. 缓存未选中，开始处理 ---
    print(f"📥 [缓存未命中] 这是一个新文档，开始完整处理流程: {file_path}")
    try:
        # A. 加载与切分
        print("   1. 加载 PDF 文件...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        print("   2. 切分文本...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        print(f"      共切分为 {len(splits)} 个片段")
        
        # B. 向量化
        print("   3. 初始化 Gitee AI 嵌入模型...")
        embeddings = GiteeAIEmbeddings(
            api_key=api_key,
            model="Qwen3-Embedding-8B",
            base_url="https://ai.gitee.com/v1"
        )
        
        print("   4. 创建 FAISS 向量存储 (这步最耗时)...")
        vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        print("      ✅ 向量库构建完成")
        
        # --- 3. 存入缓存 ---
        VECTOR_STORE_CACHE[file_path] = vectorstore
        print(f"💾 [已缓存] 文档已存入全局内存，下次提问将秒回！")
        
        return vectorstore

    except Exception as e:
        print(f"❌ 向量库创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_llm():
    """辅助函数：获取 LLM 实例"""
    return ChatOpenAI(
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="ep-20251122233041-rpp9j",
        temperature=0.1
    )


def ask_document(file_path, query, api_key):
    """
    基于 PDF 文档回答问题 (普通版)
    """
    try:
        # 1. 获取向量库 (智能缓存版)
        vectorstore = get_vectorstore(file_path, api_key)
        
        # 2. 检索
        print("🔍 正在检索相关片段...")
        relevant_docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # 3. 生成回答
        print("🤖 正在生成回答...")
        llm = get_llm()
        
        prompt = ChatPromptTemplate.from_template("""                
        你是一个智能文档问答助手，基于提供的文档内容和对话历史回答用户问题。
        
        请严格基于以下文档内容回答问题，不要添加任何超出文档的信息：
        
        {context}
        
        对话历史和用户最新问题：
        {query}
        
        回答：
        """)
        
        messages = prompt.format_messages(context=context, query=query)
        response = llm.invoke(messages)
        
        return response.content
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise


def ask_document_stream(file_path, query, api_key):
    """
    基于 PDF 文档回答问题 (流式版)
    """
    try:
        # 1. 获取向量库 (智能缓存版)
        vectorstore = get_vectorstore(file_path, api_key)
        
        # 2. 检索
        relevant_docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # 3. 生成回答 (流式)
        # 专门创建一个流式的 LLM 对象
        llm_stream = ChatOpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model="ep-20251122233041-rpp9j",
            temperature=0.1,
            streaming=True
        )
        
        prompt = ChatPromptTemplate.from_template("""                
        你是一个智能文档问答助手，基于提供的文档内容和对话历史回答用户问题。
        
        请严格基于以下文档内容回答问题，不要添加任何超出文档的信息：
        
        {context}
        
        对话历史和用户最新问题：
        {query}
        
        回答：
        """)
        
        messages = prompt.format_messages(context=context, query=query)
        
        for chunk in llm_stream.stream(messages):
            if chunk.content:
                yield chunk.content
        
    except Exception as e:
        print(f"流式生成错误: {str(e)}")
        yield f"❌ 出错啦：{str(e)}"

if __name__ == "__main__":
    print("这是后端模块，请运行 app.py")
