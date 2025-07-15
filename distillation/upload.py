from huggingface_hub import HfApi, HfFolder, login, create_repo

def upload_checkpoint():
    # Set paths and repo ID
    checkpoint_path = "results/checkpoint-14304"
    checkpoint_path = "results"
    logs_path = "logs"
    repo_id = "Pingsz/distilled-llama-1B"

    # Initialize API
    api = HfApi()
    
    # Create the repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload the checkpoint folder
    api.upload_folder(
        folder_path=checkpoint_path,
        repo_id=repo_id,
        # path_in_repo=checkpoint_path.split("/")[-1], # optionally push to a subfolder
    )
    print(f"Checkpoint folder successfully uploaded to {repo_id}")

    # Upload the logs folder
    api.upload_folder(
        folder_path=logs_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="logs",
    )
    print(f"Logs folder successfully uploaded to {repo_id}")


if __name__ == "__main__":
    login()
    upload_checkpoint()