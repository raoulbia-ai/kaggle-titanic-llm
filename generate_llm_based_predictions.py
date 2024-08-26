from openai import OpenAI
import pandas as pd
import json
import os

# Load the formatted JSONL test data (with PassengerId)
with open('data/test.jsonl', 'r') as jsonl_file:
    test_data = [json.loads(line) for line in jsonl_file]

# Initialize the OpenAI client
client = OpenAI()

# Loop through each prompt and get predictions
predictions = []
for entry in test_data:
    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::A0UkfuPZ",  # Replace with your fine-tuned model ID
        messages=entry['messages']
    )
    
    # Extract the predicted response
    predicted_response = response.choices[0].message.content.lower()
    survived = 1 if "survived" in predicted_response else 0
    predictions.append({"PassengerId": entry["PassengerId"], "Survived": survived})

# Convert predictions to a DataFrame and save to submission file
submission = pd.DataFrame(predictions)
submission.to_csv('data/submission.csv', index=False)
print("Submission file saved as data/submission.csv")
