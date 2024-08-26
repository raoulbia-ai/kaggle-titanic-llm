import openai
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

def fine_tune_model(training_file):
    fine_tune_response = openai.FineTune.create(
        model="gpt-3.5-turbo",
        training_file=open(training_file, "rb"),
    )
    fine_tune_id = fine_tune_response["id"]
    print(f"Started fine-tuning job with ID: {fine_tune_id}")

    while True:
        status = openai.FineTune.retrieve(fine_tune_id)
        if status["status"] == "succeeded":
            print("Fine-tuning completed.")
            return status["fine_tuned_model"]
        elif status["status"] == "failed":
            raise Exception("Fine-tuning failed.")
        print(f"Status: {status['status']}. Waiting...")
        time.sleep(30)  # Check status every 30 seconds

fine_tuned_model_name = fine_tune_model("data/train.jsonl")
