"""
Embeddings 模块：实现 Gitee AI Qwen3-Embedding-8B 封装
兼容 LangChain 1.0 的 Embeddings 接口
"""

import os
import requests
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
load_dotenv()
class GiteeAIEmbeddings(Embeddings):
    """
    Gitee AI 平台的 Qwen3-Embedding-8B 向量化封装
    兼容 LangChain 1.0 的 Embeddings 接口
    
    参考文档: https://ai.gitee.com/docs/openapi/v1#tag/%E7%89%B9%E5%BE%81%E6%8A%BD%E5%8F%96/post/embeddings
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://ai.gitee.com/v1",
        model: str = "Qwen3-Embedding-8B",
        dimensions: Optional[int] = None,
        default_headers: Optional[dict] = None,
    ):
        """
        初始化 GiteeAIEmbeddings
        
        Args:
            api_key: Gitee AI API 密钥
            base_url: API 基础地址
            model: 使用的模型名称
            dimensions: 向量维度
            default_headers: 默认请求头
        """
        # 优先使用传入的 api_key，如果没有则从环境变量获取
        self.api_key = api_key or os.getenv("GITEE_AI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "需要设置 GITEE_AI_API_KEY 环境变量或传入 api_key 参数"
            )
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.dimensions = dimensions
        self.default_headers = default_headers or {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 确保 Authorization 头已设置
        if "Authorization" not in self.default_headers:
            self.default_headers["Authorization"] = f"Bearer {self.api_key}"
        
        print(f"🔧 GiteeAIEmbeddings 初始化成功")
        print(f"   模型: {self.model}")
        print(f"   API 地址: {self.base_url}")
        if self.dimensions:
            print(f"   向量维度: {self.dimensions}")
    
    def _embedding_request(self, input_texts: List[str]) -> List[List[float]]:
        """
        发送嵌入请求
        
        Args:
            input_texts: 输入文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": self.model,
            "input": input_texts
        }
        
        # 添加可选的 dimensions 参数
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        
        # 发送请求
        response = requests.post(
            url=url,
            json=payload,
            headers=self.default_headers,
            timeout=30
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        
        # 提取向量
        embeddings = []
        for item in result.get("data", []):
            embedding = item.get("embedding")
            if embedding:
                embeddings.append(embedding)
        
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成向量
        
        Args:
            texts: 文档文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        return self._embedding_request(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询生成向量
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 向量
        """
        return self._embedding_request([text])[0]
