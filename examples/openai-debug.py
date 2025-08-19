import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI



async def main():
    load_dotenv()
    client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    #stream=True,
    # AsyncOpenAI

    response = await client.responses.create(
        model="gpt-4o",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
        stream=True,
    )
    async for event in response:
        print(event.type)
        if event.type == "response.completed":
            responseDone = event

    response2 = await client.responses.create(
        model="gpt-4o",
        input="and how do i do that in go?",
        previous_response_id=responseDone.response.id,
        stream=True,
    )
    async for event in response2:
        print(event)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    asyncio.run(main())