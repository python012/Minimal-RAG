"""Streamlit RAG 知识助手前端界面

此文件实现了聊天界面，并调用后端 `RAGEngine` 完成以下步骤：
1. 加载环境变量，获取模型名称和向量数据库路径
2. 页面初始化（标题/图标/布局）
3. 初始化并缓存引擎实例和会话聊天历史
4. 侧边栏：引擎重载、数据库统计、检索参数、清除历史、使用提示
5. 主区域：按顺序渲染历史消息，来源内容可折叠查看
6. 用户输入后，执行 RAG：生成嵌入 -> 相似度搜索 -> 构建上下文 -> 调用模型生成答案
7. 底部显示当前使用的 Azure OpenAI 模型和 ChromaDB 信息

"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine

# ====================== 环境变量加载 ======================
# 提前加载 .env 以确保 CHROMA_DB_PATH / 模型配置等可用

# 加载环境变量
load_dotenv()

# ====================== 页面配置 ======================
# 设置标题、图标和宽屏布局
st.set_page_config(page_title="RAG 知识助手", page_icon="🤖", layout="wide")

# ====================== 会话状态初始化 ======================
# chat_history: [{role, content, sources?}]
# rag_engine: 缓存的 RAGEngine 单例
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None


@st.cache_resource
def initialize_rag_engine():
    """构建并缓存 RAGEngine 单例

    使用 `@st.cache_resource` 避免重复初始化（连接向量数据库等耗时操作），
    仅在首次运行或手动清除缓存后重建。
    发生异常时在 UI 中显示错误消息。
    """
    db_path = os.getenv("CHROMA_DB_PATH", "./vector_db")
    try:
        engine = RAGEngine(db_path=db_path)
        return engine
    except Exception as e:
        st.error(f"初始化 RAG 引擎时出错：{e}")
        return None


# ====================== 页面标题 ======================
# 显示 Logo + 标题 + 简要描述
# st.image("static/robot.png", width=48)
st.title("📚 RAG 知识助手")
st.markdown("提出问题，基于您的知识库获取答案")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")

    # 引擎重载：清除缓存并重新运行，适合添加新数据后刷新
    if st.button("🔄 重新加载引擎"):
        st.cache_resource.clear()
        st.session_state.rag_engine = None
        st.rerun()

    # 延迟初始化：首次运行或重载后执行，带进度指示器
    if st.session_state.rag_engine is None:
        with st.spinner("正在初始化 RAG 引擎..."):
            st.session_state.rag_engine = initialize_rag_engine()

    # 引擎健康状态和向量数据库统计
    if st.session_state.rag_engine:
        st.success("✅ 引擎已加载")
        stats = st.session_state.rag_engine.get_stats()
        st.markdown(f"**数据库中的文档数：** :red[{stats['total_documents']}]")

        # 显示模型信息
        st.divider()
        st.subheader("🤖 模型配置")
        gen_model = os.getenv("AZURE_DEPLOYMENT_NAME_CHAT", "gpt-4-mini")
        embed_model = os.getenv("AZURE_DEPLOYMENT_NAME_EMBED", "text-embedding-3-small")
        st.info(f"**服务提供商：** Azure OpenAI")
        st.info(f"**生成模型：**\n`{gen_model}`")
        st.info(f"**嵌入模型：**\n`{embed_model}`")

        # 检索参数：控制结果数量和最小相关度阈值
        st.divider()
        st.subheader("🔍 搜索设置")
        n_results = st.slider("结果数量", 1, 10, 5)
        min_relevance = st.slider("使用知识库的最小相关度", 0.0, 1.0, 0.35, 0.05)

        # 来源过滤（占位符，后续可扩展实际数据源标签）
        filter_source = st.selectbox("按来源过滤", ["全部", "file", "git", "jira", "confluence"])
        if filter_source == "全部":
            filter_source = None
    else:
        st.error("❌ 引擎未加载")
        st.stop()

    st.divider()

    # 清除对话：重置历史记录，不重建引擎
    if st.button("🗑️ 清除对话历史"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # 使用说明和操作提示
    st.markdown(
        """
    ### 📚 使用方法：
    1. 通过 `data_loader.py` 加载文档
    2. 提出具体的问题
    3. 检查来源以验证准确性
    
    ### 💡 提示：
    - 问题越具体，检索效果越好
    - 使用来源展开器审查相关性
    - 添加新数据后需重新加载引擎
    """
    )

# 主聊天界面
if st.session_state.rag_engine is None:
    st.warning("⚠️ RAG 引擎未加载，请点击侧边栏中的重新加载引擎按钮重试。")
    st.stop()

# ====================== 聊天历史渲染 ======================
# 循环遍历并显示消息；折叠来源以避免主界面杂乱
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 查看来源"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**来源 {i}** (相关度：{source['relevance_score']:.2%})")
                    # 文本内容
                    st.text(source["content"])
                    # 如果存在图片路径则显示缩略图
                    img_path = source.get("metadata", {}).get("image_path")
                    if img_path:
                        st.image(img_path, width=160)
                    if source.get("metadata"):
                        st.caption(f"文件：{source['metadata'].get('file_name', '未知')}")
                    st.divider()

# 聊天输入
if prompt := st.chat_input("向您的知识库提问..."):
    # ====================== 用户输入处理 ======================
    # 1. 写入用户消息
    # 2. 渲染用户气泡
    # 3. 调用 RAG 引擎检索并生成答案，附加来源
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 使用 RAG 引擎查询方法（内部处理相关度检查）
            print(f"\n{'='*60}")
            print(f"🔍 处理查询：{prompt}")
            print(f"   最小相关度阈值：{min_relevance}")
            print(f"{'='*60}")
            
            # 获取搜索结果并检查相关度
            search_results = st.session_state.rag_engine.search(
                query=prompt,
                n_results=n_results,
                filter_source=filter_source
            )
            
            documents = search_results["documents"][0] if search_results["documents"] else []
            metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
            distances = search_results["distances"][0] if search_results["distances"] else []
            
            # 检查相关度
            max_relevance = 0.0
            if distances:
                max_relevance = max(1 - d for d in distances)
            
            is_irrelevant = (not documents) or (max_relevance < max(0.0, min_relevance))
            
            print(f"   检索到的距离：{distances[:3] if distances else '无'}")
            print(f"   最大相关度分数：{max_relevance:.6f}")
            print(f"   相关度高于阈值（命中）：{max_relevance >= max(0.0, min_relevance)}")
            
            if is_irrelevant:
                print("   ℹ️  未找到相关文档 - 使用模型的原生知识")
            else:
                print(f"   ✅ 找到 {len(documents)} 个相关文档 - 使用知识库")
            
            # 使用生成器流式输出答案
            full_response = ""
            
            # 使用 RAG 引擎的 generate_answer 方法（支持阿里云 API）
            # 当 is_irrelevant=True 时，传递空的 context_docs，引擎将使用模型的原生知识
            answer_gen = st.session_state.rag_engine.generate_answer(
                query=prompt,
                context_docs=documents if not is_irrelevant else [],
                stream=True
            )
            
            # 流式显示答案
            if answer_gen:
                for chunk in answer_gen:
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + " ⏳")
            else:
                full_response = "错误：生成答案失败"
            
            # 最终显示，不带加载指示器
            message_placeholder.markdown(full_response)
            
            print(f"✅ 答案已生成（{len(full_response)} 个字符）")
            print(f"{'='*60}\n")
            
            # 准备显示来源
            sources = []
            if not is_irrelevant:
                sources = [
                    {
                        "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        "metadata": meta,
                        "relevance_score": 1 - dist,
                    }
                    for doc, meta, dist in zip(documents, metadatas, distances)
                ]
            
            # 如果找到来源则显示
            if sources:
                with st.expander("📚 查看来源"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**来源 {i}** (相关度：{source['relevance_score']:.2%})")
                        st.text(source["content"])
                        img_path = source.get("metadata", {}).get("image_path")
                        if img_path:
                            st.image(img_path, width=160)
                        if source.get("metadata"):
                            st.caption(f"文件：{source['metadata'].get('file_name', '未知')}")
                        st.divider()
                
                # 显示顶部图片
                top_img = sources[0].get("metadata", {}).get("image_path")
                if top_img:
                    st.image(top_img, width=240)
            
            # 保存到聊天历史
            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_response, "sources": sources}
            )
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            err_msg = str(e)
            print(f"❌ 错误：{err_msg}")
            st.error(f"❌ API 错误：\n{err_msg}")
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "sources": []})

# 页脚
st.divider()
# ====================== 页脚模型信息 ======================
# 显示当前使用的模型和 ChromaDB 描述
model_info = f"Azure OpenAI ({os.getenv('AZURE_DEPLOYMENT_NAME_CHAT', 'gpt-4-mini')})"
st.caption(f"🤖 由 {model_info} 和 ChromaDB 驱动")
