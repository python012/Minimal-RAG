"""
RAG 引擎 - 检索增强生成的核心逻辑（阿里云 Qwen 版本）

什么是 RAG？
- RAG = 检索增强生成（Retrieval-Augmented Generation）
- 结合两个步骤：信息检索和文本生成
  1. 检索：从知识库中找到最相关的文档
  2. 生成：基于检索到的文档生成答案

为什么需要 RAG？
- 大语言模型（LLM）虽然强大，但知识有限（训练数据截止日期）
- RAG 让 AI 可以访问最新、私有和领域专用知识
- 答案可验证并可引用来源，更加可靠
"""

from typing import List, Dict, Optional, Any
import os
import re
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

# 首先加载 .env 以确保环境变量在模块初始化时可用
load_dotenv()

# 阿里云 API 配置
ALIYUN_MODEL_API_KEY = os.getenv("ALIYUN_MODEL_API_KEY", "")
ALIYUN_BASE_URL = os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
ALIYUN_CHAT_MODEL = os.getenv("ALIYUN_CHAT_MODEL", "qwen-plus")
ALIYUN_EMBED_MODEL = os.getenv("ALIYUN_EMBED_MODEL", "text-embedding-v4")
ALIYUN_EMBED_DIM = int(os.getenv("ALIYUN_EMBED_DIM", "1024"))
# API 超时时间（秒），默认 3 分钟
ALIYUN_API_TIMEOUT = float(os.getenv("ALIYUN_API_TIMEOUT", "180"))


class RAGEngine:
    """
    RAG 引擎类

    核心功能：
    1. 将用户问题转换为向量（Embedding）
    2. 在向量数据库中检索最相关的文档
    3. 使用检索到的文档作为上下文
    4. 调用大语言模型基于上下文生成答案

    工作流程示例：
        用户问题："如何制作某甜品？"
        ↓
        1. 向量化： [0.23, -0.45, ...]
        ↓
        2. 检索知识库：找到若干相关文档
        ↓
        3. 构建提示词：基于文档与问题
        ↓
        4. 调用模型：生成答案
        ↓
        5. 返回：答案 + 文档来源
    """

    def __init__(self, db_path: str = "./vector_db"):
        """
        Initialize RAG Engine

        Args:
            db_path: Vector database path, default ./vector_db
                    This path should be consistent with the path used in data_loader.py
        """
        # 初始化 ChromaDB 客户端（持久化存储）
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建 knowledge_base 集合
        # 若数据库不存在，将自动创建一个空集合
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base", metadata={"description": "RAG 知识库"}
        )

        # 打印初始化信息
        print("✓ RAG 引擎已初始化")
        print(f"  - 生成模型：{ALIYUN_CHAT_MODEL}（阿里云通义千问）")
        print(f"  - 嵌入模型：{ALIYUN_EMBED_MODEL}（阿里云通义千问）")
        print(f"  - 数据库路径：{db_path}")

    def get_embedding(self, text: str) -> List[float]:
        """
        使用阿里云 Qwen API 生成文本的向量表示（embedding）。

        参数：
            text: 要生成向量的文本

        返回：
            向量列表（浮点数列表）
        """
        if not ALIYUN_MODEL_API_KEY:
            raise ValueError("❌ 未配置阿里云 API 密钥，请在 .env 文件中设置 ALIYUN_MODEL_API_KEY")

        try:
            client = OpenAI(
                api_key=ALIYUN_MODEL_API_KEY,
                base_url=ALIYUN_BASE_URL,
            )
            response = client.embeddings.create(
                model=ALIYUN_EMBED_MODEL, input=text, dimensions=ALIYUN_EMBED_DIM, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 阿里云嵌入 API 失败：{e}")
            raise

    def _normalize_text(self, text: str) -> str:
        """
        规范化文本用于匹配：保留中文/英文字母和数字，移除标点符号，压缩空白符。

        示例：
        - "芒果-慕斯！" → "芒果 慕斯"
        - "Coal-Ball Ice Cream!" → "coal ball ice cream"
        """
        import re

        # 转换英文字母为小写（中文无影响）
        text = text.lower()

        # 移除标点符号，保留：中文字符、英文字母、数字、空白符
        # \u4e00-\u9fa5 覆盖基本中文汉字范围
        text = re.sub(r"[^\u4e00-\u9fa5a-z0-9\s]", " ", text)

        # 压缩多个空白符为单个空格
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _boost_exact_matches(self, query: str, results: Any) -> Any:
        """
        混合检索优化：通过精确匹配菜谱名称来提升结果排名。

        策略：
        1. 规范化用户查询（移除标点符号，转换为小写）
        2. 检查每个结果的 metadata.name 字段
        3. 如果菜谱名称与查询完全匹配或高度相关，显著降低距离分数（提升排名）

        参数：
            query: 用户查询
            results: 原始 ChromaDB 检索结果

        返回：
            优化后的结果（距离分数已调整，排名已更新）
        """
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return results

        normalized_query = self._normalize_text(query)
        query_words = set(normalized_query.split())

        # 遍历每个结果，计算精确匹配分数
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        boosted_results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            name = meta.get("name", "")
            if not name:
                boosted_results.append((doc, meta, dist, 0))
                continue

            normalized_name = self._normalize_text(name)
            name_words = set(normalized_name.split())

            # 计算匹配分数
            boost_score = 0

            # 完全匹配：将距离降至接近 0（最高优先级）
            if normalized_query == normalized_name:
                boost_score = 1000
            # 查询是名称的子串，或名称是查询的子串
            elif normalized_query in normalized_name or normalized_name in normalized_query:
                boost_score = 500
            # 词级匹配：计算重叠比例
            elif query_words and name_words:
                overlap = len(query_words & name_words)
                union = len(query_words | name_words)
                if overlap > 0:
                    boost_score = int(300 * (overlap / union))

            # 应用权重：距离越小 = 越相关，boost_score 越高 = 距离越小
            # 原始距离 * (1 - boost_factor)，boost_factor 范围 0-0.99
            if boost_score > 0:
                boost_factor = min(0.99, boost_score / 1000)
                adjusted_dist = dist * (1 - boost_factor)
            else:
                adjusted_dist = dist

            boosted_results.append((doc, meta, adjusted_dist, boost_score))

        # 按调整后的距离重新排序
        boosted_results.sort(key=lambda x: x[2])

        # 重构结果格式
        results["documents"] = [[item[0] for item in boosted_results]]
        results["metadatas"] = [[item[1] for item in boosted_results]]
        results["distances"] = [[item[2] for item in boosted_results]]

        return results

    def search(self, query: str, n_results: int = 5, filter_source: Optional[str] = None) -> Any:
        """
        在知识库中搜索与问题最相关的文档（混合检索：向量相似度 + 精确匹配加权）

        这是 RAG 的第二步：检索相关文档

        搜索原理：
        1. 将问题转换为向量：query_embedding
        2. 计算问题向量与数据库中所有文档向量的距离
        3. 距离越小 = 越相关
        4. **新增**：如果查询包含甜品名称关键词，提升精确匹配结果
        5. 返回距离最小的 n_results 个文档

        使用的距离计算方法：
        - 通常是余弦相似度或欧氏距离
        - ChromaDB 默认使用余弦相似度

        混合检索优化：
        - 当用户查询"煤球冰淇淋"时，name 字段为"煤球冰淇淋"的文档将优先返回
        - 即使向量相似度不是最高，精确匹配也会提升排名

        参数：
            query: 用户的问题，例如 "如何制作提拉米苏？"
            n_results: 返回最相关的文档数量，默认 5
                      文档越多 = 越全面但可能有噪音
                      文档越少 = 越精确但可能遗漏信息
            filter_source: 按数据源过滤，例如 "file" 或 "git"
                          None 表示不过滤，搜索所有来源

        返回：
            搜索结果字典，包含：
            {
                'documents': [[doc1, doc2, ...]],  # 检索到的文档内容
                'metadatas': [[meta1, meta2, ...]],  # 文档元数据（文件名等）
                'distances': [[dist1, dist2, ...]]   # 相似度距离（越小 = 越相关，已应用加权）
            }
        """
        try:
            # 1. 将问题转换为向量
            query_embedding = self.get_embedding(query)

            # 2. 构建过滤条件（如果指定了 filter_source）
            where_filter: Optional[Dict] = None
            if filter_source:
                where_filter = {"source": filter_source}

            # 3. 在向量数据库中搜索（检索更多候选以便精确匹配加权后重新排序）
            # 对于小数据集，搜索更多候选以确保精确匹配被召回
            search_n = min(max(n_results * 5, 30), 100)  # 搜索更多候选，最多 100
            results = self.collection.query(
                query_embeddings=[query_embedding],  # 问题向量
                n_results=search_n,  # 返回的候选数量
                where=where_filter,  # type: ignore  # 过滤条件
                include=["documents", "metadatas", "distances"],  # 返回的内容
            )

            # 4. 应用精确匹配加权优化
            results = self._boost_exact_matches(query, results)

            # 5. 裁剪到最终所需的 n_results
            if results and results.get("documents"):
                results["documents"] = [results["documents"][0][:n_results]]
                results["metadatas"] = [results["metadatas"][0][:n_results]]
                results["distances"] = [results["distances"][0][:n_results]]

            return results

        except Exception as e:
            print(f"❌ 搜索失败：{e}")
            raise

    def generate_answer(
        self,
        query: str,
        context_docs: List[str],
        chat_history: Optional[List[Dict]] = None,
        stream: bool = False,
    ):
        """
        使用阿里云 Qwen API 基于检索到的文档生成答案。

        参数：
            query: 用户问题
            context_docs: 检索到的上下文文档列表
            chat_history: 对话历史（可选，暂未使用）
            stream: 是否使用流式输出

        返回：
            生成的答案字符串，或流式生成器
        """
        if not ALIYUN_MODEL_API_KEY:
            raise ValueError("❌ 未配置阿里云 API 密钥，请在 .env 文件中设置 ALIYUN_MODEL_API_KEY")

        # 1. 检查是否有上下文文档
        has_context = context_docs and any(doc.strip() for doc in context_docs)

        # 2. 构建消息内容
        if has_context:
            context = "\n\n".join([f"文档 {i+1}：\n{doc}" for i, doc in enumerate(context_docs)])
            user_content = f"""
请基于以下上下文回答问题。
如果上下文没有提供足够的信息，请如实说明。

上下文：
{context}

问题：{query}
"""
        else:
            user_content = f"请使用你自己的知识回答以下问题：\n\n问题：{query}"

        # 3. 调用阿里云 Qwen API
        try:
            client = OpenAI(
                api_key=ALIYUN_MODEL_API_KEY,
                base_url=ALIYUN_BASE_URL,
            )
            if not stream:
                completion = client.chat.completions.create(
                    model=ALIYUN_CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个有帮助的AI助手。"},
                        {"role": "user", "content": user_content},
                    ],
                )
                return completion.choices[0].message.content
            else:
                # 流式模式
                stream_resp = client.chat.completions.create(
                    model=ALIYUN_CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个有帮助的AI助手。"},
                        {"role": "user", "content": user_content},
                    ],
                    stream=True,
                )

                def qwen_stream_generator():
                    for chunk in stream_resp:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return qwen_stream_generator()
        except Exception as e:
            print(f"❌ 阿里云对话 API 失败：{e}")
            raise

    def query(
        self,
        question: str,
        n_results: int = 5,
        filter_source: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        min_relevance: float = 0.0,
    ) -> Dict:
        """
        完整的 RAG 查询工作流（主入口方法）

        此方法整合了 RAG 的三个步骤：
        1. 搜索：查找相关文档
        2. 生成答案：基于文档生成答案
        3. 包装结果：将答案和来源一起返回

        完整工作流：
            用户问题
            ↓
            转换为向量
            ↓
            搜索知识库 → 找到 3 个文档
            ↓
            如果找到文档：
                ↓
                使用文档作为上下文
                ↓
                调用模型生成答案
                ↓
                返回：答案 + 文档来源

            如果没有找到文档：
                ↓
                让模型使用自己的知识回答
                ↓
                返回：答案 + 注明"来自模型的原生知识库"

        参数：
            question: 用户的问题
            n_results: 检索的文档数量
            filter_source: 过滤数据源
            chat_history: 对话历史（保留）
            min_relevance: 最小相关度阈值

        返回：
            结果字典：
            {
                'answer': "答案文本",
                'sources': [                    # 答案来源列表
                    {
                        'content': "文档片段",   # 文档内容（前 200 字符）
                        'metadata': {...},       # 元数据（文件名等）
                        'relevance_score': 0.85  # 相关度分数（0-1）
                    },
                    ...
                ],
                'raw_results': {...}            # 原始搜索结果
            }
        """
        try:
            # 步骤 1：搜索相关文档
            search_results = self.search(query=question, n_results=n_results, filter_source=filter_source)

            # 提取搜索结果
            documents = search_results["documents"][0] if search_results["documents"] else []
            metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
            distances = search_results["distances"][0] if search_results["distances"] else []

            # 步骤 2：检查是否找到相关文档或相关度低于阈值
            # 相关度 = 1 - 距离；比较最大相关度与阈值
            max_relevance = 0.0
            if distances:
                try:
                    max_relevance = max(1 - d for d in distances)
                    # 调试输出：查看实际距离和相关度
                    print(f"\n🔍 检索分析：")
                    print(f"   问题：{question}")
                    print(f"   前 3 个距离：{distances[:3]}")
                    print(f"   前 3 个相关度分数（1-距离）：{[round(1-d, 6) for d in distances[:3]]}")
                    print(f"   最大相关度分数：{max_relevance:.6f}")
                    print(f"   相关度阈值：{min_relevance:.6f}")
                    print(f"   满足阈值：{max_relevance >= max(0.0, min_relevance)}")
                except Exception:
                    max_relevance = 0.0

            is_irrelevant = (not documents) or (max_relevance < max(0.0, min_relevance))

            if is_irrelevant:
                # 情况 A：未找到相关文档
                # 让模型使用自己的原生知识回答
                if not ALIYUN_MODEL_API_KEY:
                    return {
                        "answer": "❌ 未配置阿里云 API 密钥，无法回答问题",
                        "sources": [],
                        "raw_results": search_results,
                    }

                try:
                    client = OpenAI(
                        api_key=ALIYUN_MODEL_API_KEY,
                        base_url=ALIYUN_BASE_URL,
                    )
                    completion = client.chat.completions.create(
                        model=ALIYUN_CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": "你是一个有帮助的AI助手。"},
                            {
                                "role": "user",
                                "content": f"请使用你自己的知识回答以下问题。不要引用知识库或编造来源：\n{question}",
                            },
                        ],
                    )
                    answer = completion.choices[0].message.content
                except Exception as e:
                    answer = f"错误：{str(e)}"

                # 返回答案，不包含任何知识库来源
                return {
                    "answer": answer,
                    "sources": [],
                    "raw_results": search_results,
                }

            # 情况 B：找到相关文档
            # 步骤 3：基于检索到的文档生成答案
            answer = self.generate_answer(query=question, context_docs=documents, chat_history=chat_history)

            # 步骤 4：组织来源信息
            # 打包文档内容、元数据和相关度分数
            sources = [
                {
                    # 仅显示前 200 个字符
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "metadata": meta,  # 文件名、块编号等
                    "relevance_score": 1 - dist,  # 距离越小 = 越相关，转换为分数（0-1）
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]

            # 步骤 5：返回完整结果
            return {"answer": answer, "sources": sources, "raw_results": search_results}

        except Exception as e:
            print(f"❌ 查询失败：{e}")
            return {"answer": f"错误：{str(e)}", "sources": [], "raw_results": None}

    def get_stats(self) -> Dict:
        """
        获取知识库统计信息

        返回：
            统计字典：
            {
                'total_documents': 150,          # 数据库中的文档块数量
                'collection_name': 'knowledge_base'  # 集合名称
            }
        """
        try:
            count = self.collection.count()
            return {"total_documents": count, "collection_name": self.collection.name}
        except Exception as e:
            print(f"❌ 获取统计信息失败：{e}")
            return {"total_documents": 0, "collection_name": "unknown"}
