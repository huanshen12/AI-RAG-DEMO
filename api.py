"""
FastAPI Hello World 示例
"""
from fastapi import FastAPI
from pydantic import BaseModel
from rag_backend import ask_document  # 导入 ask_document 函数

# 定义数据模型
class ChatRequest(BaseModel):
    """聊天请求数据模型"""
    file_path: str  # PDF 文件的路径
    query: str      # 用户的问题
    api_key: str    # Gitee API Key

# 创建 FastAPI 应用实例
app = FastAPI(
    title="Hello World API",
    description="一个简单的 FastAPI 示例",
    version="1.0.0"
)

# 定义根路径的 GET 请求处理
@app.get("/")
def read_root():
    """根路径，返回 Hello World 消息"""
    return {"message": "Hello World!", "info": "这是一个 FastAPI 示例"}

# 定义 /items/{item_id} 路径的 GET 请求处理
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    """带参数的路径，返回项目信息"""
    return {"item_id": item_id, "q": q}

# 定义 /chat 路径的 POST 请求处理
@app.post("/chat")
def chat(request: ChatRequest):
    """
    基于 PDF 文档回答问题
    
    Args:
        request: 聊天请求数据，包含 file_path、query 和 api_key
    
    Returns:
        包含回答的字典
    """
    try:
        # 调用 ask_document 函数
        answer = ask_document(
            file_path=request.file_path,
            query=request.query,
            api_key=request.api_key
        )
        # 返回回答
        return {"answer": answer}
    except Exception as e:
        # 处理异常
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
