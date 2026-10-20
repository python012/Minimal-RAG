import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ALIYUN_MODEL_API_KEY"),
    base_url="https://maas.qianwenaiapi.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # Model list：https://www.qianwenai.com/models
    model="qwen3.8-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "introduce yourself briefly."},
    ]
)
print(completion.model_dump_json())