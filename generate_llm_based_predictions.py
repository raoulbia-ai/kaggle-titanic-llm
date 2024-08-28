from openai import OpenAI
import pandas as pd
import json
import os
from tqdm import tqdm
from config import TEST_OUTPUT, SUBMISSION_FILENAME

fp=TEST_OUTPUT


# fine-tuned model ID
# model="ft:gpt-4o-mini-2024-07-18:personal::A0UkfuPZ"
# model="ft:gpt-4o-mini-2024-07-18:personal::A0X2Bb1O"
# model="ft:gpt-4o-mini-2024-07-18:personal::A0Yhsxov"
# model="ft:gpt-4o-mini-2024-07-18:personal::A0o9v5aY"  # iter 3 (Day 2, added socio-economic features)
# model="ft:gpt-4o-mini-2024-07-18:personal::A0phAiIu"  # iter 4 trained on balanced dataset and gpt-4o-mini
# model="ft:gpt-4o-2024-08-06:personal::A0qcQRkL"  # iter 5 trained on balanced dataset and gpt-4o
model="ft:gpt-4o-mini-2024-07-18:personal::A0wPhdQr"  # iter 6 Claude Sonnet v1
model="ft:gpt-4o-mini-2024-07-18:personal::A1DsHj9N"  # claude v2


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

# Wrap the main loop with tqdm for a progress bar
for entry in tqdm(test_data, desc="Processing predictions", unit="passenger"):
    try:
        response = client.chat.completions.create(
            model=model,  # Make sure to uncomment and set the correct model above
            messages=entry['messages']
        )

        predicted_response = response.choices[0].message
        survived = parse_survival_prediction(predicted_response)

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
