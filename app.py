import os
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

topic = input("Enter a topic: ")

prompt = f"""
Create 5 study flashcards about {topic}.
For each flashcard:
1. Write a question.
2. Write a short answer.
Keep the answers suitable for a 2nd-year
computer science student.
"""

start_time = time.time()

completion = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

end_time = time.time()
latency = end_time - start_time

print("\nGenerated Flashcards:\n")
print(completion.choices[0].message.content)
print(f"\nResponse time: {latency:.2f} seconds")