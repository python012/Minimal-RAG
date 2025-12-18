import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# 创建 Azure OpenAI 客户端
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# 使用 Azure 的文本嵌入模型
response = client.embeddings.create(
    model=os.getenv("AZURE_DEPLOYMENT_NAME_EMBED", "text-embedding-3-small"),
    input='the quick brown fox jumps over the lazy dog',
)

print(response.model_dump_json())
