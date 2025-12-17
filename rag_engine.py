"""
RAG Engine - Core logic for Retrieval-Augmented Generation (Ollama version)

What is RAG?
- RAG = Retrieval-Augmented Generation
- Combines two steps: information retrieval and text generation
  1. Retrieval: Find the most relevant documents from the knowledge base
  2. Generation: Generate answers based on the retrieved documents

Why do we need RAG?
- Large Language Models (LLMs) are powerful but have limited knowledge (training data cutoff)
- RAG allows AI to access the latest, private, and domain-specific knowledge
- Answers are verifiable and can cite sources, making them more reliable
"""

from typing import List, Dict, Optional, Any
import os
import requests
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

# Load .env first to ensure environment variables are available during module initialization
load_dotenv()

# Read Ollama configuration from environment variables (fallback only)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen:14b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", OLLAMA_MODEL)

# Aliyun OpenAI-compatible API config (primary)
GENERATION_MODEL_API_KEY = os.getenv("GENERATION_MODEL_API_KEY", "") or os.getenv("ALIYUN_MODEL_API_KEY", "")
ALIYUN_BASE_URL = os.getenv(
    "ALIYUN_COMPATIBLE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
ALIYUN_CHAT_MODEL = os.getenv("ALIYUN_CHAT_MODEL", "qwen-flash")
ALIYUN_EMBED_MODEL = os.getenv("ALIYUN_EMBED_MODEL", "text-embedding-v4")
ALIYUN_EMBED_DIM = int(os.getenv("ALIYUN_EMBED_DIM", "1024"))
# API timeout in seconds (default 3 minutes)
ALIYUN_API_TIMEOUT = float(os.getenv("ALIYUN_API_TIMEOUT", "180"))


class RAGEngine:
    """
    RAG Engine Class

    Core Functions:
    1. Convert user questions into vectors (embeddings)
    2. Search for the most relevant documents in the vector database
    3. Use the retrieved documents as context
    4. Call the LLM to generate answers based on context

    Workflow Example:
        User question: "How to make Mojito?"
        ↓
        1. Convert to vector: [0.23, -0.45, ...]
        ↓
        2. Search knowledge base: Find 3 relevant documents
        ↓
        3. Build prompt: "Answer based on the following documents:\nDoc1...\nDoc2...\nQuestion: How to make Mojito?"
        ↓
        4. Call model: Generate answer
        ↓
        5. Return: Answer + source documents
    """

    def __init__(self, db_path: str = "./vector_db"):
        """
        Initialize RAG Engine

        Args:
            db_path: Vector database path, default ./vector_db
                    This path should be consistent with the path used in data_loader.py
        """
        # Initialize ChromaDB client (persistent storage)
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # Get or create knowledge_base collection
        # If the database doesn't exist, an empty collection will be created automatically
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base", metadata={"description": "RAG Knowledge Base"}
        )
        
        # Print initialization info
        if GENERATION_MODEL_API_KEY:
            print("✓ RAG Engine initialized")
            print(f"  - Generation model: {ALIYUN_CHAT_MODEL} (Aliyun DashScope)")
            print(f"  - Embedding model: {ALIYUN_EMBED_MODEL} (Aliyun DashScope)")
            print(f"  - Database path: {db_path}")
        else:
            print("✓ RAG Engine initialized")
            print(f"  - Generation model: {OLLAMA_MODEL} (Local Ollama)")
            print(f"  - Embedding model: {OLLAMA_EMBED_MODEL} (Local Ollama)")
            print(f"  - Ollama URL: {OLLAMA_BASE_URL}")
            print(f"  - Database path: {db_path}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate vector representation (embedding) of text using Aliyun OpenAI-compatible API.
        Falls back to Ollama if Aliyun config is missing.
        """
        # Prefer Aliyun OpenAI-compatible API when configured
        if GENERATION_MODEL_API_KEY:
            try:
                client = OpenAI(
                    api_key=GENERATION_MODEL_API_KEY,
                    base_url=ALIYUN_BASE_URL,
                    timeout=ALIYUN_API_TIMEOUT,  # Configurable timeout
                )
                completion = client.embeddings.create(
                    model=ALIYUN_EMBED_MODEL,
                    input=text,
                    dimensions=ALIYUN_EMBED_DIM,
                    encoding_format="float",
                )
                return completion.data[0].embedding
            except Exception as e:
                print(f"❌ Aliyun embedding API failed: {e}")
                raise

        # Fallback: use local Ollama if Aliyun key not set
        try:
            url = f"{OLLAMA_BASE_URL}/api/embeddings"
            payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.HTTPError as e:
            err_msg = str(e)
            if response is not None:
                try:
                    err_msg += f"\nOllama response: {response.text}"
                except Exception:
                    pass
            if "memory" in err_msg.lower() or "out of memory" in err_msg.lower():
                raise RuntimeError(f"Ollama out of memory or model too large: {err_msg}")
            raise RuntimeError(f"Ollama API error: {err_msg}")
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            raise

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching: convert to lowercase, remove special characters, compress whitespace.

        Example: "Apple-Jack!" -> "apple jack"
        """
        import re

        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _boost_exact_matches(self, query: str, results: Any) -> Any:
        """
        Hybrid retrieval optimization: boost results with exact recipe name matches.

        Strategy:
        1. Normalize user query (remove punctuation, convert to lowercase)
        2. Check metadata.name field for each result
        3. If recipe name matches query exactly or is highly related, significantly reduce distance score (boost ranking)

        Args:
            query: User query
            results: Raw ChromaDB retrieval results

        Returns:
            Optimized results (distance scores adjusted, ranking updated)
        """
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return results

        normalized_query = self._normalize_text(query)
        query_words = set(normalized_query.split())

        # Iterate through each result and calculate exact match score
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

            # Calculate match score
            boost_score = 0

            # Exact match: reduce distance to near 0 (highest priority)
            if normalized_query == normalized_name:
                boost_score = 1000
            # Query is substring of name or name is substring of query
            elif normalized_query in normalized_name or normalized_name in normalized_query:
                boost_score = 500
            # Word-level match: calculate overlap ratio
            elif query_words and name_words:
                overlap = len(query_words & name_words)
                union = len(query_words | name_words)
                if overlap > 0:
                    boost_score = int(300 * (overlap / union))

            # Apply weighting: smaller distance = more relevant, higher boost_score = smaller distance
            # Original distance * (1 - boost_factor), boost_factor range 0-0.99
            if boost_score > 0:
                boost_factor = min(0.99, boost_score / 1000)
                adjusted_dist = dist * (1 - boost_factor)
            else:
                adjusted_dist = dist

            boosted_results.append((doc, meta, adjusted_dist, boost_score))

        # Re-sort by adjusted distance
        boosted_results.sort(key=lambda x: x[2])

        # Reconstruct result format
        results["documents"] = [[item[0] for item in boosted_results]]
        results["metadatas"] = [[item[1] for item in boosted_results]]
        results["distances"] = [[item[2] for item in boosted_results]]

        return results

    def search(self, query: str, n_results: int = 5, filter_source: Optional[str] = None) -> Any:
        """
        Search for documents most relevant to the question in the knowledge base (hybrid retrieval: vector similarity + exact match weighting)

        This is the second step of RAG: retrieve relevant documents

        Search principle:
        1. Convert question to vector: query_embedding
        2. Calculate distance between question vector and all document vectors in database
        3. Smaller distance = more relevant
        4. **New addition**: If query contains recipe name keywords, boost exact match results
        5. Return n_results documents with smallest distances

        Distance calculation method used:
        - Usually Cosine Similarity or Euclidean Distance
        - ChromaDB defaults to cosine similarity

        Hybrid retrieval optimization:
        - When user queries "Apple Jack", documents with name field "Apple Jack" will be returned first
        - Even if vector similarity is not highest, exact match will boost ranking

        Args:
            query: User's question, e.g., "How to make Mojito?"
            n_results: Number of most relevant documents to return, default 5
                      More documents = more comprehensive but may have noise
                      Fewer documents = more precise but may miss information
            filter_source: Filter by data source, e.g., "file" or "git"
                          None means no filter, search all sources

        Returns:
            Search result dictionary containing:
            {
                'documents': [[doc1, doc2, ...]],  # Retrieved document content
                'metadatas': [[meta1, meta2, ...]],  # Document metadata (filenames, etc.)
                'distances': [[dist1, dist2, ...]]   # Similarity distances (smaller = more relevant, with weighting applied)
            }
        """
        try:
            # 1. Convert question to vector
            query_embedding = self.get_embedding(query)

            # 2. Build filter condition (if filter_source is specified)
            where_filter: Optional[Dict] = None
            if filter_source:
                where_filter = {"source": filter_source}

            # 3. Search in vector database (retrieve more candidates for re-ranking after exact match weighting)
            # For small datasets, search more candidates to ensure exact matches are recalled
            search_n = min(max(n_results * 5, 30), 100)  # Search more candidates, max 100
            results = self.collection.query(
                query_embeddings=[query_embedding],  # Question vector
                n_results=search_n,  # Number of candidates to return
                where=where_filter,  # type: ignore  # Filter condition
                include=["documents", "metadatas", "distances"],  # Content to return
            )

            # 4. Apply exact match weighting optimization
            results = self._boost_exact_matches(query, results)

            # 5. Trim to final n_results needed
            if results and results.get("documents"):
                results["documents"] = [results["documents"][0][:n_results]]
                results["metadatas"] = [results["metadatas"][0][:n_results]]
                results["distances"] = [results["distances"][0][:n_results]]

            return results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            raise

    def generate_answer(
        self, query: str, context_docs: List[str], chat_history: Optional[List[Dict]] = None, stream: bool = False
    ):
        """
        Generate answer based on retrieved documents using Aliyun OpenAI-compatible API.
        Falls back to Ollama if Aliyun config is missing.
        """
        # 1. Check if we have context documents
        has_context = context_docs and any(doc.strip() for doc in context_docs)
        
        # 2. Build message content
        if has_context:
            context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
            user_content = f"""
Please answer the question based on the following context.
If the context doesn't provide sufficient information, please state so honestly.

Context:
{context}

Question: {query}
"""
        else:
            user_content = f"Please answer the following question using your own knowledge:\n\nQuestion: {query}"

        # 3. Call Aliyun OpenAI-compatible chat API
        if GENERATION_MODEL_API_KEY:
            try:
                client = OpenAI(
                    api_key=GENERATION_MODEL_API_KEY,
                    base_url=ALIYUN_BASE_URL,
                    timeout=ALIYUN_API_TIMEOUT,  # Configurable timeout
                )
                if not stream:
                    completion = client.chat.completions.create(
                        model=ALIYUN_CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": user_content},
                        ],
                    )
                    return completion.choices[0].message.content
                else:
                    # Streaming mode
                    stream_resp = client.chat.completions.create(
                        model=ALIYUN_CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": user_content},
                        ],
                        stream=True,
                    )
                    def aliyun_stream_generator():
                        for chunk in stream_resp:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    return aliyun_stream_generator()
            except Exception as e:
                print(f"❌ Aliyun chat API failed: {e}")
                raise

        # Fallback: use local Ollama
        try:
            import json
            # Build Ollama-style prompt
            if has_context:
                context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
                prompt = f"""
You are a helpful AI assistant. Please answer the question based on the following context.
If the context doesn't provide sufficient information, please state so honestly.

Context:
{context}

Question: {query}
"""
            else:
                prompt = f"""Please answer the following question using your own knowledge:\n\nQuestion: {query}"""

            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": stream}
            
            if not stream:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()["response"]
            else:
                response = requests.post(url, json=payload, timeout=120, stream=True)
                response.raise_for_status()
                def ollama_stream_generator():
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                            except json.JSONDecodeError:
                                continue
                return ollama_stream_generator()
        except requests.exceptions.HTTPError as e:
            err_msg = str(e)
            try:
                if hasattr(e, 'response') and e.response is not None:
                    err_msg += f"\nOllama response: {e.response.text}"
            except Exception:
                pass
            if "memory" in err_msg.lower() or "out of memory" in err_msg.lower():
                raise RuntimeError(f"Ollama out of memory or model too large: {err_msg}")
            raise RuntimeError(f"Ollama API error: {err_msg}")
        except Exception as e:
            print(f"❌ Failed to generate answer: {e}")
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
        Complete RAG query workflow (main entry method)

        This method integrates the three steps of RAG:
        1. Search: Find relevant documents
        2. Generate answer: Generate answer based on documents
        3. Wrap results: Return answer and sources together

        Complete workflow:
            User question
            ↓
            Convert to vector
            ↓
            Search knowledge base → Find 3 documents
            ↓
            If documents found:
                ↓
                Use documents as context
                ↓
                Call model to generate answer
                ↓
                Return: Answer + document sources

            If no documents found:
                ↓
                Let model answer using its own knowledge
                ↓
                Return: Answer + note "from model's native knowledge base"

        Args:
            question: User's question
            n_results: Number of documents to retrieve
            filter_source: Filter data source
            chat_history: Chat history (reserved)

        Returns:
            Result dictionary:
            {
                'answer': "Answer text",
                'sources': [                    # List of answer sources
                    {
                        'content': "Document snippet",   # Document content (first 200 chars)
                        'metadata': {...},       # Metadata (filename, etc.)
                        'relevance_score': 0.85  # Relevance score (0-1)
                    },
                    ...
                ],
                'raw_results': {...}            # Raw search results
            }
        """
        try:
            # Step 1: Search for relevant documents
            search_results = self.search(query=question, n_results=n_results, filter_source=filter_source)

            # Extract search results
            documents = search_results["documents"][0] if search_results["documents"] else []
            metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
            distances = search_results["distances"][0] if search_results["distances"] else []

            # Step 2: Check if relevant documents are found or relevance is below threshold
            # relevance = 1 - distance; Compare max relevance with threshold
            max_relevance = 0.0
            if distances:
                try:
                    max_relevance = max(1 - d for d in distances)
                    # Debug output: view actual distances and relevance
                    print(f"\n🔍 Retrieval Analysis:")
                    print(f"   Question: {question}")
                    print(f"   Top 3 distances: {distances[:3]}")
                    print(f"   Top 3 relevance scores (1-distance): {[round(1-d, 6) for d in distances[:3]]}")
                    print(f"   Max relevance score: {max_relevance:.6f}")
                    print(f"   Relevance threshold: {min_relevance:.6f}")
                    print(f"   Meets threshold: {max_relevance >= max(0.0, min_relevance)}")
                except Exception:
                    max_relevance = 0.0

            is_irrelevant = (not documents) or (max_relevance < max(0.0, min_relevance))

            if is_irrelevant:
                # Case A: No relevant documents found
                # Let model answer using its own native knowledge
                if GENERATION_MODEL_API_KEY:
                    try:
                        client = OpenAI(
                            api_key=GENERATION_MODEL_API_KEY,
                            base_url=ALIYUN_BASE_URL,
                            timeout=ALIYUN_API_TIMEOUT,  # Configurable timeout
                        )
                        completion = client.chat.completions.create(
                            model=ALIYUN_CHAT_MODEL,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": f"Please answer the following question using your own knowledge. Do not cite knowledge base or fabricate sources:\n{question}"},
                            ],
                        )
                        answer = completion.choices[0].message.content
                    except Exception as e:
                        answer = f"Error: {str(e)}"
                else:
                    # Fallback to Ollama
                    prompt = f"Please answer the following question using your own knowledge. Do not cite knowledge base or fabricate sources:\n{question}"
                    url = f"{OLLAMA_BASE_URL}/api/generate"
                    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
                    try:
                        response = requests.post(url, json=payload, timeout=120)
                        response.raise_for_status()
                        answer = response.json()["response"]
                    except Exception as e:
                        answer = f"Error: {str(e)}"

                # Return answer without any knowledge base sources
                return {
                    "answer": answer,
                    "sources": [],
                    "raw_results": search_results,
                }

            # Case B: Relevant documents found
            # Step 3: Generate answer based on retrieved documents
            answer = self.generate_answer(query=question, context_docs=documents, chat_history=chat_history)

            # Step 4: Organize source information
            # Package document content, metadata, and relevance scores
            sources = [
                {
                    # Only display first 200 characters
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "metadata": meta,  # Filename, chunk number, etc.
                    "relevance_score": 1 - dist,  # Smaller distance = more relevant, convert to score (0-1)
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]

            # Step 5: Return complete results
            return {"answer": answer, "sources": sources, "raw_results": search_results}

        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {"answer": f"Error: {str(e)}", "sources": [], "raw_results": None}

    def get_stats(self) -> Dict:
        """
        Get knowledge base statistics

        Returns:
            Statistics dictionary:
            {
                'total_documents': 150,          # Number of document chunks in database
                'collection_name': 'knowledge_base'  # Collection name
            }
        """
        try:
            count = self.collection.count()
            return {"total_documents": count, "collection_name": self.collection.name}
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            return {"total_documents": 0, "collection_name": "unknown"}
