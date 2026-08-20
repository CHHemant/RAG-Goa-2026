import os
import json
from openai import OpenAI


API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set."
    )


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


MODEL = "openrouter/free"


response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a precise Hindi question-answering assistant. "
                "Answer the question clearly and concisely in Hindi."
            )
        },
        {
            "role": "user",
            "content": "मैनहट्टन परियोजना क्या थी?"
        }
    ],
    temperature=0.1,
    max_tokens=300
)


print()
print("=" * 80)
print("OPENROUTER RAW RESPONSE")
print("=" * 80)

print(
    json.dumps(
        response.model_dump(),
        ensure_ascii=False,
        indent=2
    )
)

print("=" * 80)