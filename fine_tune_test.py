from openai import OpenAI
client = OpenAI()

# uploading
# client.files.create(
#   file=open("data/train.jsonl", "rb"),
#   purpose="fine-tune"
# )

# finetune
client.fine_tuning.jobs.create(
  training_file="file-WalHl7Wwf8ymspyEEsPVGbK7", 
  model="gpt-4o-mini-2024-07-18"
)

# # List 10 fine-tuning jobs
# client.fine_tuning.jobs.list(limit=10)

# Retrieve the state of a fine-tune
# client.fine_tuning.jobs.retrieve("ftjob-fLBLS23RSyGD7g6j7UfEAV5Q")

# # Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-fLBLS23RSyGD7g6j7UfEAV5Q")

# # List up to 10 events from a fine-tuning job
# client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-fLBLS23RSyGD7g6j7UfEAV5Q", limit=10)

# # Delete a fine-tuned model (must be an owner of the org the model was created in)
# client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:fLBLS23RSyGD7g6j7UfEAV5Q")