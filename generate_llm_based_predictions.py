from openai import OpenAI
import pandas as pd
import json
import os
from tqdm import tqdm
from config import TEST_OUTPUT, SUBMISSION_FILENAME, MODEL_RESPONSES

fp = TEST_OUTPUT

model="gpt-4o-mini-2024-07-18"  # baseline model

# fine-tuned model ID
# model = "ft:gpt-4o-mini-2024-07-18:personal::A0wPhdQr"  # claude v2

def parse_survival_prediction(response):
    response_text = str(response).lower()
    if "survived" in response_text and "did not survive" not in response_text:
        return 1
    elif "did not survive" in response_text:
        return 0
    else:
        return None

# Load the formatted JSONL test data
with open(fp, 'r') as jsonl_file:
    test_data = [json.loads(line) for line in jsonl_file]

# Initialize the OpenAI client
client = OpenAI()

# Loop through each prompt and get predictions
predictions = []
unclear_predictions = []
model_responses = []  # New list to store model responses

# Wrap the main loop with tqdm for a progress bar
for entry in tqdm(test_data, desc="Processing predictions", unit="passenger"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=entry['messages']
        )

        predicted_response = response.choices[0].message
        survived = parse_survival_prediction(predicted_response)

        # Store the model's response
        model_responses.append({
            "PassengerId": entry["PassengerId"],
            "ModelResponse": predicted_response.content
        })

        if survived is not None:
            predictions.append({"PassengerId": entry["PassengerId"], "Survived": survived})
        else:
            print(f"Unclear prediction for PassengerId {entry['PassengerId']}: {predicted_response}")
            unclear_predictions.append(entry["PassengerId"])
            print(f"Unclear prediction for PassengerId {entry['PassengerId']}")

    except Exception as e:
        print(f"Error processing PassengerId {entry['PassengerId']}: {str(e)}")
        unclear_predictions.append(entry["PassengerId"])

# Handle unclear predictions
if unclear_predictions:
    print(f"There were {len(unclear_predictions)} unclear predictions.")
    for pid in unclear_predictions:
        predictions.append({"PassengerId": pid, "Survived": 0})

# Ensure the submissions directory exists
os.makedirs('data/submissions', exist_ok=True)

# Convert predictions to a DataFrame and save to submission file
submission = pd.DataFrame(predictions)
submission.to_csv(f'{SUBMISSION_FILENAME}', index=False)
print(f"Submission file saved as {SUBMISSION_FILENAME} with {len(submission)} predictions")

# Save model responses to a JSON file
responses_filename = MODEL_RESPONSES
with open(responses_filename, 'w') as f:
    json.dump(model_responses, f, indent=2)
print(f"Model responses saved to {responses_filename}")