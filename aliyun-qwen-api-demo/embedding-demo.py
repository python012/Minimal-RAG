import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ALIYUN_MODEL_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.embeddings.create(
    model="text-embedding-v4",
    input='the quick brown fox jumps over the lazy dog',
    dimensions=1024, # specify embedding dimension
    encoding_format="float"
)

print(completion.model_dump_json())
