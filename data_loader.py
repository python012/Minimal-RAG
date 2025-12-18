"""
数据加载器 - 将文档加载到向量数据库
支持文本分块、批量导入、统计查询等功能
使用阿里云 Qwen 模型生成向量嵌入
"""

import os
import sys
import glob
import argparse
import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

# 首先加载 .env 以确保环境变量可用
load_dotenv()

# 阿里云 API 配置
ALIYUN_MODEL_API_KEY = os.getenv("ALIYUN_MODEL_API_KEY", "")
ALIYUN_BASE_URL = os.getenv(
    "ALIYUN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
ALIYUN_EMBED_MODEL = os.getenv("ALIYUN_EMBED_MODEL", "text-embedding-v4")
ALIYUN_EMBED_DIM = int(os.getenv("ALIYUN_EMBED_DIM", "1024"))
# API 超时时间（秒），默认 3 分钟
ALIYUN_API_TIMEOUT = float(os.getenv("ALIYUN_API_TIMEOUT", "180"))


class DataLoader:
    """
    数据加载器类

    功能：
    1. 从文本文件读取内容
    2. 将长文本分割成小块
    3. 为每个文本块生成向量表示（embedding）
    4. 将向量和文本存储到 ChromaDB 向量数据库

    使用场景：
    - 导入技术文档、产品手册、常见问题
    - 构建企业知识库
    - 为 RAG 问答系统准备数据源
    """

    def __init__(self, db_path: str = "./vector_db"):
        """
        初始化数据加载器

        参数：
            db_path: 向量数据库存储路径，默认为 ./vector_db
        """
        # 初始化 ChromaDB 客户端（持久化存储）
        # ChromaDB 是一个专门用于存储和检索嵌入向量的轻量级向量数据库
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)  # 禁用匿名遥测
        )

        # 获取或创建名为 "knowledge_base" 的集合
        # 集合类似于传统数据库中的"表"
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base", metadata={"description": "RAG 知识库"}
        )

        print("✓ 数据加载器已初始化")
        print(f"  - 嵌入模型：{ALIYUN_EMBED_MODEL}（阿里云通义千问）")
        print(f"  - 向量维度：{ALIYUN_EMBED_DIM}")
        print(f"  - API 端点：{ALIYUN_BASE_URL}")
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
                model=ALIYUN_EMBED_MODEL,
                input=text,
                dimensions=ALIYUN_EMBED_DIM,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 阿里云嵌入 API 失败：{e}")
            raise

    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        将长文本分割成带有一定重叠的小块

        为什么要分割文本？
        1. 向量数据库对每个文本块有大小限制
        2. 小块语义更集中，检索更准确
        3. 避免无关内容干扰检索结果

        什么是重叠？
        - 相邻的块之间有一些重叠的内容
        - 例如：chunk1="...ABC"，chunk2="BC..."，BC 就是重叠部分
        - 重叠可以防止关键信息被切分到两个块之间

        分割策略：
        1. 优先在句号（.）处分割，保持句子完整
        2. 其次在换行符（\n）处分割
        3. 避免分割点过早（> chunk_size // 2）

        参数：
            text: 要分割的长文本
            chunk_size: 每个块的大小（字符数），默认 1000
            overlap: 相邻块之间的重叠大小（字符数），默认 200

        返回：
            文本块列表，例如 ["第一部分...", "第二部分...", ...]
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # 计算当前块的结束位置
            end = start + chunk_size
            chunk = text[start:end]

            # 如果不是文本末尾，尝试在句子边界处分割
            if end < text_len:
                # 查找最后一个句号的位置
                last_period = chunk.rfind(".")
                # 查找最后一个换行符的位置
                last_newline = chunk.rfind("\n")
                # 选择较后的位置作为分割点
                break_point = max(last_period, last_newline)

                # 只有在分割点不是太早的情况下才使用（避免块太小）
                if break_point > chunk_size // 2:
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            # 去除首尾空白并添加到结果列表
            chunks.append(chunk.strip())

            # 下一个块的开始位置 = 当前结束位置 - 重叠大小
            # 这样就创建了相邻块之间 'overlap' 个字符的重叠
            start = end - overlap

        return chunks

    def load_text_file(self, file_path: str, source_type: str = "file") -> int:
        """
        将单个文本文件加载到向量数据库

        完整工作流程：
        1. 读取文件内容
        2. 分割成多个块
        3. 为每个块生成嵌入向量
        4. 存储到 ChromaDB，并带上元数据（文件名、块编号等）

        参数：
            file_path: 文件路径，例如 "./data/sample.txt"
            source_type: 数据源类型标签，例如 "file"、"git"、"jira"
                        用于后续过滤检索（例如只搜索 Git 文档）

        返回：
            成功加载的块数量
        """
        try:
            # 1. 读取文件内容（使用 UTF-8 编码）
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 2. 将内容分割成块
            chunks = self.split_text(content)

            print(f"📄 处理文件：{file_path}")
            print(f"   - 文件大小：{len(content)} 字符")
            print(f"   - 分割结果：{len(chunks)} 块")

            # 3. 为每个块生成嵌入向量并存储
            for i, chunk in enumerate(chunks):
                # 跳过空块
                if not chunk.strip():
                    continue

                # 生成嵌入向量（调用阿里云 API）
                print(f"   - 处理块 {i+1}/{len(chunks)}...", end="\r")
                embedding = self.get_embedding(chunk)

                # 存储到 ChromaDB
                # embeddings: 向量列表
                # documents: 原始文本列表
                # metadatas: 元数据列表（文件名、来源、块编号等）
                # ids: 唯一标识符列表（由文件路径 + 编号组成）
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source": source_type,  # 数据源类型
                            "file_path": file_path,  # 完整文件路径
                            "file_name": os.path.basename(file_path),  # 文件名
                            "chunk_id": i,  # 块编号（从 0 开始）
                            "total_chunks": len(chunks),  # 此文件的总块数
                            "chunk_size": len(chunk),  # 当前块的大小
                        }
                    ],
                    ids=[f"{file_path}_{i}"],  # 唯一 ID，例如 "./data/doc.txt_0"
                )

            print(f"   ✓ 完成：{len(chunks)} 块已导入")
            return len(chunks)

        except OSError as e:
            print(f"❌ 文件读取错误 {file_path}：{e}")
            return 0
        except UnicodeDecodeError as e:
            print(f"❌ 文件编码错误 {file_path}：{e}")
            print("   提示：请确保文件使用 UTF-8 编码")
            return 0
        except Exception as e:
            print(f"❌ 文件处理失败 {file_path}：{e}")
            return 0

    def _match_image_for_recipe(
        self, recipe_basename: str, preferred_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        在 `data/images/` 中匹配与食谱文件同名（不含扩展名）的图片。

        支持常见扩展名：.jpg / .jpeg / .png
        返回匹配的相对路径（例如 data/images/xxx.jpg），或者 None（未匹配）。
        """
        # 支持多个图片目录和显式文件名
        image_dirs = [
            os.path.join("data", "images"),
            os.path.join("data2", "images"),
        ]

        # 如果 JSON 提供了显式文件名，首先尝试匹配它
        filenames: List[str] = []
        if preferred_filename and preferred_filename.strip():
            base = preferred_filename.strip()
            filenames.append(base)

        # 然后尝试按同名规则匹配
        for ext in [".jpg", ".jpeg", ".png"]:
            filenames.append(recipe_basename + ext)

        for d in image_dirs:
            if not os.path.isdir(d):
                continue
            for fn in filenames:
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    return p
        return None

    def load_json_recipe_file(self, file_path: str, source_type: str = "file") -> int:
        """
        将单个 JSON 甜品文件加载到向量数据库。

        解析字段并生成语义文本用于嵌入；自动匹配同名图片路径并写入元数据。

        支持的 JSON 字段（中文字段）：
        - 名称 / name：甜品名称（必需）
        - 描述 / description：甜品描述
        - 类型 / type：甜品类型（如"冰淇淋"、"布丁"等）
        - 关键词 / keywords / tags：标签数组或逗号分隔字符串
        - 配料 / ingredients：配料列表，元素为 {数量, 单位, 原料} 的字典
        - 步骤 / directions / instructions：制作步骤（字符串或字符串列表）
        - 图片 / image：图片文件名
        - 来源 / source：资料来源
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 标准化字段（中文优先）
            name = str(
                data.get("名称") or data.get("name") or os.path.splitext(os.path.basename(file_path))[0]
            )
            description = data.get("描述") or data.get("description") or ""
            recipe_type = data.get("类型") or data.get("type") or ""

            # 关键词字段
            tags = data.get("关键词") or data.get("keywords") or data.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            # 配料字段（仅支持中文格式）
            ingredients = data.get("配料") or data.get("ingredients") or []
            ing_lines: List[str] = []
            if isinstance(ingredients, list):
                for ing in ingredients:
                    if isinstance(ing, dict):
                        qty = str(ing.get("数量") or ing.get("quantity") or "").strip()
                        measure = str(ing.get("单位") or ing.get("measure") or "").strip()
                        ingredient_name = str(ing.get("原料") or ing.get("ingredient") or "").strip()
                        line = " ".join([p for p in [qty, measure, ingredient_name] if p])
                        if line:
                            ing_lines.append(f"- {line}")
                    elif isinstance(ing, str):
                        ing_lines.append(f"- {ing.strip()}")

            # 步骤字段（仅支持中文格式）
            instructions = data.get("步骤") or data.get("directions") or data.get("instructions") or []
            instr_lines: List[str] = []
            if isinstance(instructions, list):
                for i, step in enumerate(instructions, 1):
                    if isinstance(step, str) and step.strip():
                        instr_lines.append(f"{i}. {step.strip()}")
            elif isinstance(instructions, str):
                steps_list = [s.strip() for s in instructions.split("|") if s.strip()]
                for i, step in enumerate(steps_list, 1):
                    instr_lines.append(f"{i}. {step}")

            # 将步骤合并为单一字符串，便于在元数据中保存与检索
            instructions_text = "\n".join(instr_lines) if instr_lines else ""

            # 组合语义文本
            text_parts = [
                f"名称：{name}",
                f"描述：{description}" if description else "",
                f"类型：{recipe_type}" if recipe_type else "",
                f"关键词：{', '.join(tags)}" if tags else "",
                "配料：" if ing_lines else "",
                *ing_lines,
                "步骤：" if instr_lines else "",
                *instr_lines,
            ]
            semantic_text = "\n".join([t for t in text_parts if t])

            # 生成嵌入向量
            embedding = self.get_embedding(semantic_text)

            # 匹配同名图片
            basename = os.path.splitext(os.path.basename(file_path))[0]
            json_image = data.get("图片") or data.get("image")
            image_path = self._match_image_for_recipe(basename, str(json_image or ""))

            # 写入向量数据库
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags or "")
            image_path_str = str(image_path or "")
            source_str = str(data.get("来源") or data.get("source") or "")

            self.collection.add(
                embeddings=[embedding],
                documents=[semantic_text],
                metadatas=[
                    {
                        "source": str(source_type or ""),
                        "file_path": str(file_path or ""),
                        "file_name": os.path.basename(file_path),
                        "name": str(name or ""),
                        "type": str(recipe_type or ""),
                        "tags": tags_str,
                        "image_path": image_path_str,
                        "recipe_source": source_str,
                        "instructions": instructions_text,
                    }
                ],
                ids=[f"{file_path}"],
            )

            print(f"   ✓ 甜品导入完成：{name}")
            return 1

        except Exception as e:
            print(f"❌ JSON 甜品处理失败 {file_path}：{e}")
            return 0

    def load_directory(
        self,
        dir_path: str,
        patterns: List[str] = None,  # type: ignore
        source_type: str = "file",
        recursive: bool = True,
    ) -> int:
        """
        批量加载目录中的所有文件

        支持的功能：
        1. 按文件扩展名过滤（例如只加载 .txt 和 .md）
        2. 递归搜索子目录
        3. 批量处理多个文件

        参数：
            dir_path: 目录路径，例如 "./data"
            patterns: 文件匹配模式列表，例如 ["*.txt", "*.md"]
                     None 时默认为 ["*.txt", "*.md"]
            source_type: 数据源类型标签
            recursive: 是否递归搜索子目录，默认 True

        返回：
            成功加载的总块数
        """
        # 如果未指定模式，使用默认模式
        if patterns is None:
            # 默认支持 txt / md / json
            patterns = ["*.txt", "*.md", "*.json"]

        total_chunks = 0

        # 遍历每个匹配模式
        for pattern in patterns:
            # 构建搜索路径
            # 递归："./data/**/*.txt"（搜索所有子目录）
            # 非递归："./data/*.txt"（只搜索当前目录）
            if recursive:
                search_pattern = f"{dir_path}/**/{pattern}"
            else:
                search_pattern = f"{dir_path}/{pattern}"

            # 使用 glob 查找所有匹配的文件
            files = glob.glob(search_pattern, recursive=recursive)

            print(f"\n🔍 搜索模式：{pattern}")
            print(f"   找到 {len(files)} 个文件")

            # 逐个处理文件：根据扩展名区分处理方法
            for file_path in files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".json":
                    added = self.load_json_recipe_file(file_path, source_type)
                    total_chunks += added
                else:
                    chunks = self.load_text_file(file_path, source_type)
                    total_chunks += chunks

        return total_chunks

    def clear_database(self):
        """
        清空数据库中的所有数据

        操作流程：
        1. 删除现有的 "knowledge_base" 集合
        2. 重新创建一个空的 "knowledge_base" 集合
        3. 插入一条占位记录以确保集合正确初始化

        使用场景：
        - 重新导入前清空旧数据
        - 切换到不同的知识库内容
        - 测试和调试

        注意：
        ⚠️ 此操作不可逆，所有已导入的数据将会丢失！
        """
        try:
            print("🗑️  正在清空数据库...")

            # 删除集合
            self.chroma_client.delete_collection("knowledge_base")

            # 重新创建集合
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base", metadata={"description": "RAG 知识库"}
            )

            # 插入一条占位记录以确保集合正确初始化
            dummy_text = "这是一条用于初始化集合的占位记录。"
            dummy_embedding = self.get_embedding(dummy_text)
            self.collection.add(
                embeddings=[dummy_embedding],
                documents=[dummy_text],
                metadatas=[
                    {
                        "source": "system",
                        "file_path": "__dummy__",
                        "file_name": "__dummy__",
                        "name": "占位符",
                        "tags": "system",
                        "glass": "",
                        "garnish": "",
                        "alcoholic": "",
                        "image_path": "",
                    }
                ],
                ids=["__dummy_init__"],
            )

            print("✓ 数据库已清空（已插入占位记录）")

        except Exception as e:
            print(f"❌ 清空数据库失败：{e}")

    def get_stats(self) -> Dict:
        """
        获取数据库统计信息

        返回：
            包含统计信息的字典：
            {
                'total_chunks': 100,           # 总块数
                'collection_name': 'knowledge_base'  # 集合名称
            }
        """
        count = self.collection.count()
        return {"total_chunks": count, "collection_name": self.collection.name}


def main():
    """
    命令行接口 (CLI)

    支持的命令：
    1. 导入单个文件：
       python data_loader.py --input ./data/doc.txt

    2. 导入整个目录：
       python data_loader.py --input ./data/

    3. 指定文件类型：
       python data_loader.py --input ./data/ --pattern *.txt *.md

    4. 导入前清空数据库：
       python data_loader.py --input ./data/ --clear

    5. 查看统计信息：
       python data_loader.py --stats
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="将文档加载到向量数据库（支持阿里云 Qwen 模型）")

    # 必需参数：输入文件或目录
    parser.add_argument("--input", "-i", help="输入文件或目录路径")

    # 可选参数：数据源类型标签
    parser.add_argument("--source", "-s", default="file", help="数据源类型（file、git、jira 等）")

    # 可选参数：文件匹配模式
    parser.add_argument(
        "--pattern",
        "-p",
        nargs="+",
        default=["*.txt", "*.md"],
        help="文件匹配模式，例如：*.txt *.md",
    )

    # 可选参数：是否清空数据库
    parser.add_argument("--clear", action="store_true", help="导入前清空数据库")

    # 可选参数：显示统计信息
    parser.add_argument("--stats", action="store_true", help="显示数据库统计信息")

    args = parser.parse_args()

    # 加载环境变量（从 .env 文件）
    load_dotenv()

    # 从环境变量读取数据库路径
    db_path = os.getenv("CHROMA_DB_PATH", "./vector_db")

    # 初始化数据加载器
    print("=" * 60)
    print("📦 阿里云 Qwen RAG 数据加载器")
    print("=" * 60)

    loader = DataLoader(db_path=db_path)

    # 如果指定了 --clear，清空数据库
    if args.clear:
        loader.clear_database()
        # 如果只是清空而没有输入，成功退出
        if not args.input and not args.stats:
            sys.exit(0)

    # 如果指定了 --stats，显示统计信息并退出
    if args.stats:
        stats = loader.get_stats()
        print("\n📊 数据库统计信息：")
        print(f"  - 总块数：{stats['total_chunks']}")
        print(f"  - 集合名称：{stats['collection_name']}")
        sys.exit(0)

    # 必须指定 --input 或 --stats（除非已经由 --clear 处理）
    if not args.input:
        print("❌ 错误：请使用 --input 指定输入文件/目录，或使用 --stats 查看统计信息")
        print("\n使用示例：")
        print("  python data_loader.py --input ./data/doc.txt")
        print("  python data_loader.py --input ./data/ --pattern *.txt")
        print("  python data_loader.py --stats")
        print("  python data_loader.py --clear")
        sys.exit(1)

    # 开始加载数据
    input_path = args.input

    if os.path.isfile(input_path):
        # 输入是单个文件
        print(f"\n📄 加载单个文件：{input_path}")
        chunks = loader.load_text_file(input_path, args.source)
        print(f"\n✅ 成功加载 {chunks} 块")

    elif os.path.isdir(input_path):
        # 输入是目录
        print(f"\n📁 加载目录：{input_path}")
        print(f"   匹配模式：{args.pattern}")
        chunks = loader.load_directory(input_path, args.pattern, args.source)
        print(f"\n✅ 成功加载 {chunks} 块")

    else:
        print(f"❌ 错误：{input_path} 不是有效的文件或目录")
        sys.exit(1)

    # 显示最终统计信息
    stats = loader.get_stats()
    print("\n" + "=" * 60)
    print("📊 最终统计信息：")
    print(f"  - 数据库中的总块数：{stats['total_chunks']}")
    print(f"  - 集合名称：{stats['collection_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
