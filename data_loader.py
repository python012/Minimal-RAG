"""
数据加载器 - 将文档加载到向量数据库
支持文本分块、批量导入、统计查询等功能
使用 Azure OpenAI API 生成向量嵌入和对话
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
from openai import AzureOpenAI

# 首先加载 .env 以确保环境变量可用
load_dotenv()

# Azure OpenAI API 配置
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT_NAME_CHAT = os.getenv("AZURE_DEPLOYMENT_NAME_CHAT", "gpt-4-mini")
AZURE_DEPLOYMENT_NAME_EMBED = os.getenv("AZURE_DEPLOYMENT_NAME_EMBED", "text-embedding-3-small")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
# API 超时时间（秒），默认 3 分钟
AZURE_API_TIMEOUT = float(os.getenv("AZURE_API_TIMEOUT", "180"))


class DataLoader:
    """
    Data Loader Class

    Features:
    1. Read content from text files
    2. Split long text into small chunks
    3. Generate vector representation (embedding) for each text chunk
    4. Store vectors and text to ChromaDB vector database

    Use cases:
    - Import technical documentation, product manuals, FAQs
    - Build enterprise knowledge base
    - Prepare data source for RAG Q&A system
    """

    def __init__(self, db_path: str = "./vector_db"):
        """
        Initialize data loader

        Args:
            db_path: Vector database storage path, default is ./vector_db
        """
        # Initialize ChromaDB client (persistent storage)
        # ChromaDB is a lightweight vector database specifically for storing and retrieving embeddings
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)  # Disable anonymous telemetry
        )

        # Get or create collection named "knowledge_base"
        # Collection is similar to "table" in traditional databases
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base", metadata={"description": "RAG Knowledge Base"}
        )

        print("✓ 数据加载器已初始化")
        print(f"  - 嵌入模型：{AZURE_DEPLOYMENT_NAME_EMBED}（Azure OpenAI）")
        print(f"  - 生成模型：{AZURE_DEPLOYMENT_NAME_CHAT}（Azure OpenAI）")
        print(f"  - API 端点：{AZURE_OPENAI_ENDPOINT}")
        print(f"  - 数据库路径：{db_path}")

    def get_embedding(self, text: str) -> List[float]:
        """
        使用 Azure OpenAI API 生成文本的向量表示（embedding）。

        参数：
            text: 要生成向量的文本

        返回：
            向量列表（浮点数列表）
        """
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            raise ValueError("❌ 未配置 Azure OpenAI 密钥或端点，请在 .env 文件中设置 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT")

        try:
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
            response = client.embeddings.create(
                model=AZURE_DEPLOYMENT_NAME_EMBED,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Azure OpenAI 嵌入 API 失败：{e}")
            raise

    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split long text into small chunks with certain overlap

        Why split text?
        1. Vector databases have size limits for each text chunk
        2. Small chunks have more focused semantics for more accurate retrieval
        3. Avoid irrelevant content interfering with retrieval results

        What is overlap?
        - Adjacent chunks have some overlapping content
        - Example: chunk1="...ABC", chunk2="BC...", BC is the overlap
        - Overlap prevents key information from being split between two chunks

        Splitting strategy:
        1. Split at period (.) first to keep sentences intact
        2. Then split at newline (\n)
        3. Avoid split points too early (> chunk_size // 2)

        Args:
            text: Long text to split
            chunk_size: Size of each chunk (character count), default 1000
            overlap: Overlap size between adjacent chunks (character count), default 200

        Returns:
            List of text chunks, e.g. ["First part...", "Second part...", ...]
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # Calculate end position of current chunk
            end = start + chunk_size
            chunk = text[start:end]

            # If not at end of text, try to split at sentence boundary
            if end < text_len:
                # Find position of last period
                last_period = chunk.rfind(".")
                # Find position of last newline
                last_newline = chunk.rfind("\n")
                # Choose the later position as split point
                break_point = max(last_period, last_newline)

                # Only use split point if not too early (avoid chunk too small)
                if break_point > chunk_size // 2:
                    chunk = chunk[: break_point + 1]
                    end = start + break_point + 1

            # Strip whitespace and add to result list
            chunks.append(chunk.strip())

            # Next chunk start position = current end position - overlap size
            # This creates overlap of 'overlap' characters between adjacent chunks
            start = end - overlap

        return chunks

    def load_text_file(self, file_path: str, source_type: str = "file") -> int:
        """
        Load single text file into vector database

        Complete workflow:
        1. Read file content
        2. Split into multiple chunks
        3. Generate embedding for each chunk
        4. Store to ChromaDB along with metadata (filename, chunk number, etc.)

        Args:
            file_path: File path, e.g. "./data/sample.txt"
            source_type: Data source type tag, e.g. "file", "git", "jira"
                        Used for filtering retrieval later (e.g. search only Git documents)

        Returns:
            Number of successfully loaded chunks
        """
        try:
            # 1. Read file content (using UTF-8 encoding)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 2. Split content into chunks
            chunks = self.split_text(content)

            print(f"📄 Processing file: {file_path}")
            print(f"   - File size: {len(content)} characters")
            print(f"   - Split result: {len(chunks)} chunks")

            # 3. Generate embedding for each chunk  and store
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue

                # Generate embedding (call Ollama API)
                print(f"   - Processing chunk {i+1}/{len(chunks)}...", end="\r")
                embedding = self.get_embedding(chunk)

                # Store to ChromaDB
                # embeddings: vector list
                # documents: original text list
                # metadatas: metadata list (filename, source, chunk number, etc.)
                # ids: unique identifier list (composed of file path + number)
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source": source_type,  # Data source type
                            "file_path": file_path,  # Full file path
                            "file_name": os.path.basename(file_path),  # Filename
                            "chunk_id": i,  # Chunk number (starts from 0)
                            "total_chunks": len(chunks),  # Total chunks in this file
                            "chunk_size": len(chunk),  # Current chunk size
                        }
                    ],
                    ids=[f"{file_path}_{i}"],  # Unique ID, e.g. "./data/doc.txt_0"
                )

            print(f"   ✓ Complete: {len(chunks)} chunks imported")
            return len(chunks)

        except OSError as e:
            print(f"❌ File read error {file_path}: {e}")
            return 0
        except UnicodeDecodeError as e:
            print(f"❌ File encoding error {file_path}: {e}")
            print("   Tip: Please ensure file is UTF-8 encoded")
            return 0
        except Exception as e:
            print(f"❌ File processing failed {file_path}: {e}")
            return 0

    def _match_image_for_recipe(
        self, recipe_basename: str, preferred_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Match image with same name as recipe file (without extension) in `data/images/`.

        Supports common extensions: .jpg / .jpeg / .png
        Returns matched relative path (e.g. data/images/xxx.jpg), or None if not matched.
        """
        # 支持多个图片目录和显式文件名
        image_dirs = [
            os.path.join("data", "images"),
            os.path.join("data2", "images"),
        ]

        # If JSON provides explicit filename, try matching with that first
        filenames: List[str] = []
        if preferred_filename and preferred_filename.strip():
            base = preferred_filename.strip()
            filenames.append(base)

        # Then try matching by same name rule
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
        Batch load all files in directory

        Supported features:
        1. Filter by file extension (e.g. load only .txt and .md)
        2. Recursively search subdirectories
        3. Batch process multiple files

        Args:
            dir_path: Directory path, e.g. "./data"
            patterns: File matching pattern list, e.g. ["*.txt", "*.md"]
                     None Defaults to ["*.txt", "*.md"]
            source_type: Data source type tag
            recursive: Whether to recursively search subdirectories, default True

        Returns:
            Successfully loadedTotal chunk count
        """
        # Use default patterns if not specified
        if patterns is None:
            # Default support for txt / md / json
            patterns = ["*.txt", "*.md", "*.json"]

        total_chunks = 0

        # Iterate through each matching pattern
        for pattern in patterns:
            # Build search path
            # Recursive: "./data/**/*.txt" (search all subdirectories)
            # Non-recursive: "./data/*.txt" (search current directory only)
            if recursive:
                search_pattern = f"{dir_path}/**/{pattern}"
            else:
                search_pattern = f"{dir_path}/{pattern}"

            # Use glob to find all matching files
            files = glob.glob(search_pattern, recursive=recursive)

            print(f"\n🔍 Search pattern: {pattern}")
            print(f"   Found {len(files)} files")

            # Process files one by one: distinguish processing method by extension
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
        Clear all data in database

        Operation workflow:
        1. Delete existing "knowledge_base" collection
        2. Recreate an empty "knowledge_base" collection
        3. Insert a dummy record to ensure collection is properly initialized

        Use cases:
        - Clear old data before reimporting
        - Switch to different knowledge base content
        - Testing and debugging

        Note:
        ⚠️ This operation is irreversible, all imported data will be lost!
        """
        try:
            print("🗑️  Clearing database...")

            # Delete collection
            self.chroma_client.delete_collection("knowledge_base")

            # Recreate collection
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base", metadata={"description": "RAG Knowledge Base"}
            )

            # Insert a dummy record to ensure collection is properly initialized
            dummy_text = "This is a placeholder record to initialize the collection."
            dummy_embedding = self.get_embedding(dummy_text)
            self.collection.add(
                embeddings=[dummy_embedding],
                documents=[dummy_text],
                metadatas=[
                    {
                        "source": "system",
                        "file_path": "__dummy__",
                        "file_name": "__dummy__",
                        "name": "Placeholder",
                        "tags": "system",
                        "glass": "",
                        "garnish": "",
                        "alcoholic": "",
                        "image_path": "",
                    }
                ],
                ids=["__dummy_init__"],
            )

            print("✓ Database cleared (with dummy record inserted)")

        except Exception as e:
            print(f"❌ Failed to clear database: {e}")

    def get_stats(self) -> Dict:
        """
        Get database statistics

        Returns:
            Dictionary containing statistics:
            {
                'total_chunks': 100,           # Total chunk count
                'collection_name': 'knowledge_base'  # collectionName
            }
        """
        count = self.collection.count()
        return {"total_chunks": count, "collection_name": self.collection.name}


def main():
    """
    Command Line Interface (CLI)

    Supported commands:
    1. Import single file:
       python data_loader.py --input ./data/doc.txt

    2. Import entire directory:
       python data_loader.py --input ./data/

    3. Specify file types:
       python data_loader.py --input ./data/ --pattern *.txt *.md

    4. Clear database before import:
       python data_loader.py --input ./data/ --clear

    5. View statistics:
       python data_loader.py --stats
    """
    # Create command line argument parser
    parser = argparse.ArgumentParser(description="Load documents into vector database (supports Ollama local models)")

    # Required argument: input file or directory
    parser.add_argument("--input", "-i", help="Input file or directory path")

    # Optional argument: Data source type tag
    parser.add_argument("--source", "-s", default="file", help="Data source type (file, git, jira,, etc.)")

    # Optional argument: file matching pattern
    parser.add_argument(
        "--pattern",
        "-p",
        nargs="+",
        default=["*.txt", "*.md"],
        help="File matching pattern, e.g.: *.txt *.md",
    )

    # Optional argument: whether to clear database
    parser.add_argument("--clear", action="store_true", help="Clear database before import")

    # Optional argument: display statistics
    parser.add_argument("--stats", action="store_true", help="Display database statistics")

    args = parser.parse_args()

    # Load environment variables (from .env file)
    load_dotenv()

    # Read from environment variablesDatabase path
    db_path = os.getenv("CHROMA_DB_PATH", "./vector_db")

    # Initialize data loader
    print("=" * 60)
    print("📦 Ollama RAG Data Loader")
    print("=" * 60)

    loader = DataLoader(db_path=db_path)

    # Clear database if --clear specified
    if args.clear:
        loader.clear_database()
        # If only clearing without input, exit successfully
        if not args.input and not args.stats:
            sys.exit(0)

    # Display statistics and exit if --stats specified
    if args.stats:
        stats = loader.get_stats()
        print("\n📊 Database Statistics:")
        print(f"  - Total chunk count: {stats['total_chunks']}")
        print(f"  - collectionName: {stats['collection_name']}")
        sys.exit(0)

    # Must specify --input or --stats (unless already handled by --clear)
    if not args.input:
        print("❌ Error: Please use --input to specify input file/directory, or use --stats to view statistics")
        print("\nUsage examples:")
        print("  python data_loader.py --input ./data/doc.txt")
        print("  python data_loader.py --input ./data/ --pattern *.txt")
        print("  python data_loader.py --stats")
        print("  python data_loader.py --clear")
        sys.exit(1)

    # Start loading data
    input_path = args.input

    if os.path.isfile(input_path):
        # Input is a single file
        print(f"\n📄 Loading single file: {input_path}")
        chunks = loader.load_text_file(input_path, args.source)
        print(f"\n✅ Successfully loaded {chunks} chunks")

    elif os.path.isdir(input_path):
        # Input is a directory
        print(f"\n📁 Loading directory: {input_path}")
        print(f"   Matching patterns: {args.pattern}")
        chunks = loader.load_directory(input_path, args.pattern, args.source)
        print(f"\n✅ Successfully loaded {chunks} chunks")

    else:
        print(f"❌ Error: {input_path} is not a valid file or directory")
        sys.exit(1)

    # Display final statistics
    stats = loader.get_stats()
    print("\n" + "=" * 60)
    print("📊 Final Statistics:")
    print(f"  - Total chunks in database: {stats['total_chunks']}")
    print(f"  - collectionName: {stats['collection_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
