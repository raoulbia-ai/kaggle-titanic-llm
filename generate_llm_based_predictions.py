from openai import OpenAI
import pandas as pd
import json
import os

def parse_survival_prediction(response):
    # Convert the ChatCompletionMessage to a string and lower case it
    response_text = str(response).lower()
    
    # Check for 'survived' or 'did not survive' anywhere in the response
    if "survived" in response_text and "did not survive" not in response_text:
        return 1
    elif "did not survive" in response_text:
        return 0
    else:
        # If we can't determine, we'll return None and handle it later
        return None

# Load the formatted JSONL test data (with PassengerId)
with open('data/test_no_nlp.jsonl', 'r') as jsonl_file:
    test_data = [json.loads(line) for line in jsonl_file]

# Initialize the OpenAI client
client = OpenAI()

# Loop through each prompt and get predictions
predictions = []
unclear_predictions = []
# model="ft:gpt-4o-mini-2024-07-18:personal::A0UkfuPZ"  # v1
# model="ft:gpt-4o-mini-2024-07-18:personal::A0X2Bb1O"  # v2
# model="ft:gpt-4o-mini-2024-07-18:personal::A0Yhsxov"  # v3
# model="ft:gpt-4o-mini-2024-07-18:personal::A0o9v5aY"  # iter 3 (Day 2, added socio-economic features)
# model = "ft:gpt-4o-mini-2024-07-18:personal::A0phAiIu"  # iter 4 trained on balanced dataset and gpt-4o-mini
model = "ft:gpt-4o-2024-08-06:personal::A0qcQRkL"  # iter 5 trained on balanced dataset and gpt-4o
for entry in test_data:
    try:
        response = client.chat.completions.create(
            model=model,  # Replace with your fine-tuned model ID
            messages=entry['messages']
        )
        
        # Extract the predicted response
        predicted_response = response.choices[0].message
        survived = parse_survival_prediction(predicted_response)
        
        if survived is not None:
            predictions.append({"PassengerId": entry["PassengerId"], "Survived": survived})
        else:
            unclear_predictions.append(entry["PassengerId"])
            print(f"Unclear prediction for PassengerId {entry['PassengerId']}")
    
    except Exception as e:
        print(f"Error processing PassengerId {entry['PassengerId']}: {str(e)}")
        unclear_predictions.append(entry["PassengerId"])

# Handle unclear predictions
if unclear_predictions:
    print(f"There were {len(unclear_predictions)} unclear predictions.")
    # You might want to implement a fallback strategy here, such as:
    # 1. Manually review these cases
    # 2. Use a default prediction (e.g., the most common outcome)
    # 3. Re-run these specific cases with a different prompt

    # For now, let's use a default prediction of 0 (did not survive)
    for pid in unclear_predictions:
        predictions.append({"PassengerId": pid, "Survived": 0})

# Convert predictions to a DataFrame and save to submission file
submission = pd.DataFrame(predictions)
submission.to_csv('data/submission_test_no_nlp_gpt4o.csv', index=False)
print(f"Submission file saved as data/submission_test_no_nlp_gpt4o.csv with {len(submission)} predictions")

# Optional: Print out some statistics
total_predictions = len(submission)
survived_count = submission['Survived'].sum()
print(f"Total predictions: {total_predictions}")
print(f"Predicted survivors: {survived_count}")
print(f"Predicted casualties: {total_predictions - survived_count}")
print(f"Survival rate: {survived_count / total_predictions:.2%}")