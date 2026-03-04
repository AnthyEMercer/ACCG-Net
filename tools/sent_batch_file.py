import sys, os
sys.path.append(os.getcwd())

from openai import OpenAI
from config import *

import httpx
insecure_client = httpx.Client(
    verify=False
)

client = OpenAI(
    api_key=SILICONFLOW, 
    base_url="https://api.siliconflow.cn/v1"
)

batch_input_file = client.files.create(
    file=open("", "rb"),
    purpose="batch"
)
print(batch_input_file)

batch_input_file_id = batch_input_file.data['id']
print(batch_input_file_id)

client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "nightly eval job"
    },
    extra_body={"replace":{"model": "deepseek-ai/DeepSeek-V3"}}
)
# input()

# while True:
#     batch = client.batches.retrieve("batch_abc123")
#     print(batch)
#     input()