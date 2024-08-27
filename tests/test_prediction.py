from openai import OpenAI
import json
import os

# fp='data/test.jsonl'
fp='data/test_refined.jsonl'
fp='data/test_refined_iter_3.jsonl'
fp='data/test_no_nlp.jsonl'

# fine-tuned model ID
# model="ft:gpt-4o-mini-2024-07-18:personal::A0UkfuPZ"
# model="ft:gpt-4o-mini-2024-07-18:personal::A0X2Bb1O"
# model="ft:gpt-4o-mini-2024-07-18:personal::A0Yhsxov"
# model="ft:gpt-4o-mini-2024-07-18:personal::A0o9v5aY"  # iter 3 (Day 2, added socio-economic features)
# model = "ft:gpt-4o-mini-2024-07-18:personal::A0phAiIu"  # iter 4 trained on balanced dataset and gpt-4o-mini
model = "ft:gpt-4o-2024-08-06:personal::A0qcQRkL"  # iter 5 trained on balanced dataset and gpt-4o

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
print("Test batch completed.")
