## Get a list of available models on academic cloud

from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()

client = OpenAI(base_url="https://chat-ai.academiccloud.de/v1")
models = [m.id for m in client.models.list()]
for idx, model in enumerate(sorted(models)):
    print(f'{idx:02d}: {model}')
