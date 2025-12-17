import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ALIYUN_MODEL_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # Model list：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "introduce yourself briefly."},
    ]
)
print(completion.model_dump_json())