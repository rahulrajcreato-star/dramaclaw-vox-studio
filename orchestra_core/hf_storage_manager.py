import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from orchestra_core.config import HF_TOKEN, HF_DEFAULT_DATASET

class HuggingFaceStorageManager:
    def __init__(self, token: str = None, repo_id: str = None):
        self.token = token or HF_TOKEN
        self.repo_id = repo_id or HF_DEFAULT_DATASET
        self.api = HfApi(token=self.token) if self.token else None

    def update_token(self, token: str, repo_id: str = ""):
        self.token = token.strip()
        if repo_id:
            self.repo_id = repo_id.strip()
        if self.token:
            self.api = HfApi(token=self.token)

    def check_connection(self) -> dict:
        if not self.token:
            return {"authenticated": False, "username": "Anonymous", "message": "No Hugging Face token set."}
        try:
            user_info = self.api.whoami()
            username = user_info.get("name", "Unknown")
            return {
                "authenticated": True,
                "username": username,
                "message": f"Connected to Hugging Face as @{username}."
            }
        except Exception as e:
            return {
                "authenticated": False,
                "username": "Error",
                "message": f"Authentication failed: {str(e)[:60]}"
            }

    def upload_master_video(self, local_video_path: str, remote_filename: str = None) -> dict:
        """
        Uploads a rendered 1080p documentary to persistent Hugging Face storage.
        """
        if not os.path.exists(local_video_path):
            return {"success": False, "error": f"Local file not found: {local_video_path}"}

        if not self.token:
            return {
                "success": False,
                "error": "Cannot upload to Hugging Face without HF_TOKEN. Saved locally in orchestra_outputs/.",
                "local_path": local_video_path
            }

        filename = remote_filename or os.path.basename(local_video_path)
        try:
            # Ensure dataset repo exists
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                    private=False
                )
            except Exception:
                pass

            # Upload file
            commit_info = self.api.upload_file(
                path_or_fileobj=local_video_path,
                path_in_repo=f"master_videos/{filename}",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Upload master video {filename} from DramaClaw Orchestra"
            )

            hf_url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/master_videos/{filename}"
            return {
                "success": True,
                "url": hf_url,
                "repo_id": self.repo_id,
                "commit": str(commit_info),
                "message": f"Master video successfully saved to Hugging Face: {hf_url}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Hugging Face upload error: {e}",
                "local_path": local_video_path
            }
