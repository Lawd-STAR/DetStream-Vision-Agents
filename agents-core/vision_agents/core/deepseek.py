

"""
There are various hosting for model providers
- baseten
- bedrock (converse API: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#)
- cerebras
- openrouter
- vertex
- fal
- modal
- ollama
- wandb
- azure AI foundry
- together AI (Together SDK)
- novita AI (openai)
- Fireworks
- Replicate
- Hyperbolic
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def main():

    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"],
        base_url="https://inference.baseten.co/v1"
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.1",
        messages=[
            {
                "role": "user",
                "content": "Implement FizzBuzz in Python"
            }
        ],
        stop=[],
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True
        },
        top_p=1,
        max_tokens=1000,
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)


if __name__ == '__main__':
    main()