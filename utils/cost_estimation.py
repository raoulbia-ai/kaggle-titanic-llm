import pandas as pd
import tiktoken
import json

# Constants for Estimation
MAX_TOKENS_PER_EXAMPLE = 16385
TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25
MODEL_NAME = "gpt-4o-mini-2024-07-18"  # Replace with your model if different

def get_token_length(messages, model=MODEL_NAME):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(message["content"])) for message in messages)
    return total_tokens

def estimate_cost(jsonl_data):
    convo_lens = [get_token_length(convo["messages"]) for convo in jsonl_data]
    
    # Calculate epochs based on the dataset size
    n_train_examples = len(convo_lens)
    n_epochs = TARGET_EPOCHS
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    # Estimate the total tokens billed
    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    total_billed_tokens = n_epochs * n_billing_tokens_in_dataset
    
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged during training.")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset.")
    print(f"Estimated tokens charged: ~{total_billed_tokens} tokens")

    return total_billed_tokens

def main():
    # Replace this with the path to your JSONL dataset
    jsonl_data_path = 'data/test.jsonl'
    with open(jsonl_data_path, 'r') as file:
        jsonl_data = [json.loads(line) for line in file]

    estimate_cost(jsonl_data)

if __name__ == "__main__":
    main()
