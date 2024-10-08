from openai import OpenAI
import json
import os
from config import TEST_OUTPUT

fp=TEST_OUTPUT


model="gpt-4o-mini-2024-07-18"  # baseline model

# fine-tuned model ID
# model="ft:gpt-4o-mini-2024-07-18:personal::A1fNulFL"  # claude v2



# Load the formatted JSONL test data (with PassengerId)
with open(fp, 'r') as jsonl_file:
    test_data = [json.loads(line) for line in jsonl_file]

# Initialize the OpenAI client
client = OpenAI()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the number of prompts to test (e.g., 5 for a quick inspection)
test_batch_size = 1



# Loop through the first few prompts and inspect the LLM responses
for i, entry in enumerate(test_data[:test_batch_size]):
    response = client.chat.completions.create(
        model=model, 
        messages=entry['messages']
    )
    
    predicted_response = response.choices[0].message
    print(f"PassengerId: {entry['PassengerId']}")
    print(f"User Prompt: {entry['messages'][1]['content']}")
    print(f"LLM Response: {predicted_response}")
    print("-" * 50)  # Separator for readability

# Optionally, print a message if you want to track when the test finishes

