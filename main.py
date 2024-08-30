import json
import pandas as pd
from data_prep import prepare_data
from prompts import generate_prompt, create_assistant_message, SYSTEM_MESSAGE
from config import TRAIN_FILE, TEST_FILE, TRAIN_OUTPUT, TEST_OUTPUT

def create_train_jsonl(data, output_file):
    with open(output_file, 'w') as f:
        for _, passenger in data.iterrows():
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE
                    },
                    {
                        "role": "user",
                        "content": generate_prompt(passenger, is_train=True)
                    },
                    {
                        "role": "assistant",
                        "content": create_assistant_message(passenger)
                    }
                ]
            }
            json.dump(entry, f)
            f.write('\n')

def create_test_jsonl(data, output_file):
    with open(output_file, 'w') as f:
        for _, passenger in data.iterrows():
            entry = {
                "PassengerId": int(passenger['PassengerId']),
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE
                    },
                    {
                        "role": "user",
                        "content": generate_prompt(passenger, is_train=False)
                    }
                ]
            }
            json.dump(entry, f)
            f.write('\n')



def main():
    # Prepare and process training data
    print("Processing training data...")
    train_data = prepare_data(TRAIN_FILE, is_train=True)
    create_train_jsonl(train_data, TRAIN_OUTPUT)
    print(f"Training data processed and saved to {TRAIN_OUTPUT}")

    # Prepare and process test data
    print("Processing test data...")
    test_data = prepare_data(TEST_FILE, is_train=False)
    create_test_jsonl(test_data, TEST_OUTPUT)
    print(f"Test data processed and saved to {TEST_OUTPUT}")

if __name__ == "__main__":
    main()