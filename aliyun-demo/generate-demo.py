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

# 使用 Azure 的 gpt-4-mini 模型进行对话
response = client.chat.completions.create(
    model=os.getenv("AZURE_DEPLOYMENT_NAME_CHAT", "gpt-4-mini"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "introduce yourself briefly."},
    ]
)
print(response.model_dump_json())