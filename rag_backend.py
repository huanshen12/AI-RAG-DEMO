# rag_backend.py
import os
import warnings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 屏蔽烦人的警告
warnings.filterwarnings("ignore")

# --- 正确的导入方式 ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import GiteeAIEmbeddings  # 使用 Gitee AI Embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# 导入必要的模块
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
print("✅ 使用直接检索 + 生成的方式")


def ask_document(file_path, query, api_key):
    """
    基于 PDF 文档回答问题
    """
    try:
        # --- 阶段 A: 数据处理 ---
        print("1. 加载 PDF 文件...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"   成功加载 {len(docs)} 页文档")
        
        # 2. 切分文本
        print("2. 切分文本...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        print(f"   成功切分为 {len(splits)} 个文本片段")
        
        # --- 阶段 B: 向量化存储 ---
        print("3. 初始化 Gitee AI 嵌入模型...")
        # 使用 Gitee AI Embeddings
        try:
            embeddings = GiteeAIEmbeddings(
                api_key=api_key,
                model="Qwen3-Embedding-8B",
                base_url="https://ai.gitee.com/v1"
            )
            print("   嵌入模型初始化成功")
        except Exception as e:
            print(f"   ❌ 嵌入模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("4. 创建向量存储...")
        try:
            # 测试单个文本的向量化
            print("   测试单个文本向量化...")
            test_vector = embeddings.embed_query("测试文本")
            print(f"   单个文本向量化成功，向量维度: {len(test_vector)}")
            
            # 测试批量文本的向量化
            print("   测试批量文本向量化...")
            test_texts = ["测试文本1", "测试文本2"]
            test_vectors = embeddings.embed_documents(test_texts)
            print(f"   批量文本向量化成功，生成了 {len(test_vectors)} 个向量")
            
            # 创建向量存储
            print("   正在创建 FAISS 向量存储...")
            vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=embeddings
            )
            print("   向量存储创建成功")
        except Exception as e:
            print(f"   ❌ 向量存储创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # --- 阶段 C: 检索与生成 ---
        print("5. 初始化大模型...")
        try:
            # 使用 Gitee AI 的大模型
            llm = ChatOpenAI(
                base_url="https://ai.gitee.com/v1",
                api_key=api_key,
                model="DeepSeek-V3",
                temperature=0.7
            )
            print("   大模型初始化成功")
        except Exception as e:
            print(f"   ❌ 大模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("6. 检索相关文档...")
        try:
            # 使用向量存储检索相关文档
            print("   正在检索相关文档...")
            # 直接使用 similarity_search 方法检索文档
            relevant_docs = vectorstore.similarity_search(query, k=3)
            print(f"   成功检索到 {len(relevant_docs)} 个相关文档")
            
            # 构建上下文
            context = "\n".join([doc.page_content for doc in relevant_docs])
            print(f"   上下文长度: {len(context)} 字符")
        except Exception as e:
            print(f"   ❌ 检索文档失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("7. 生成回答...")
        try:
            # 创建提示模板
            prompt = ChatPromptTemplate.from_template("""
            你是一个智能文档问答助手，基于提供的文档内容回答用户问题。
            
            请严格基于以下文档内容回答问题，不要添加任何超出文档的信息：
            
            {context}
            
            用户问题：{query}
            
            回答：
            """)
            
            # 生成回答
            print("   正在生成回答...")
            # 直接使用大模型生成回答
            messages = prompt.format_messages(context=context, query=query)
            # 使用 invoke 方法调用大模型
            response = llm.invoke(messages)
            answer = response.content
            
            print("   回答生成成功")
            print(f"   回答内容: {answer[:100]}...")
        except Exception as e:
            print(f"   ❌ 回答生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return answer
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("这是后端模块，请运行 app.py")
