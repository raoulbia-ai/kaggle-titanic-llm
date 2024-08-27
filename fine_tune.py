from openai import OpenAI
import time

client = OpenAI()

def upload_file(file_path):
    print(f"Uploading file: {file_path}")
    try:
        with open(file_path, "rb") as file:
            upload_response = client.files.create(
                file=file,
                purpose="fine-tune"
            )
        print(f"File uploaded successfully. File ID: {upload_response.id}")
        return upload_response.id
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return None

def get_latest_file_id(filename):
    print(f"Retrieving file ID for: {filename}")
    try:
        response = client.files.list()
        # Filter files by purpose and filename, then sort by creation time (descending)
        relevant_files = [
            f for f in response.data 
            if f.purpose == "fine-tune" and f.filename == filename
        ]
        if not relevant_files:
            print(f"No file found with name: {filename}")
            return None
        
        latest_file = max(relevant_files, key=lambda x: x.created_at)
        print(f"Latest file ID retrieved: {latest_file.id}")
        return latest_file.id
    except Exception as e:
        print(f"Error retrieving file ID: {str(e)}")
        return None

def wait_for_file_processing(file_id):
    print("Waiting for file to be processed...")
    while True:
        file_status = client.files.retrieve(file_id)
        if file_status.status == "processed":
            print("File processing completed.")
            return True
        elif file_status.status == "error":
            print("Error in file processing.")
            return False
        print("File still processing. Waiting...")
        time.sleep(10)  # Wait for 10 seconds before checking again

def start_fine_tuning(file_id, model):
    print(f"Starting fine-tuning job with file ID: {file_id}")
    try:
        job = client.fine_tuning.jobs.create(
            training_file=file_id, 
            model=model
        )
        print(f"Fine-tuning job created successfully. Job ID: {job.id}")
        return job.id
    except Exception as e:
        print(f"Error starting fine-tuning job: {str(e)}")
        return None

def main():

    base_dir = "data/train/jsonl"
    FILENAME = "train_refined_iter_3_balanced.jsonl"
    file_path = f"{base_dir}/{FILENAME}"
    
    model = "gpt-4o-mini-2024-07-18"
    # model = "gpt-4o-2024-08-06"  # EXPENSIVE!!!


    # Upload file
    upload_file(file_path)

    # Get the latest file ID
    file_id = get_latest_file_id(FILENAME)
    if not file_id:
        return
    print(file_id)
    
    # Wait for file processing
    if not wait_for_file_processing(file_id):
        return

    # Start fine-tuning
    job_id = start_fine_tuning(file_id, model)
    if job_id:
        print(f"Fine-tuning job started. You can monitor its progress using the job ID: {job_id}")

if __name__ == "__main__":
    main()