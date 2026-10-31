#!/usr/bin/env python3
"""
================================================================================
DRAMACLAW HUGGING FACE SPACES AUTONOMOUS CLOUD DEPLOYER
================================================================================
Deploys the DramaClaw Studio directly to Hugging Face Spaces so that it stays
online 24/7 in the cloud even when your local PC is shut down!

Usage:
    python deploy_to_hf_space.py --token hf_yourTokenHere
================================================================================
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def deploy(token: str = None, space_name: str = "dramaclaw-vox-studio"):
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("\n" + "=" * 70)
        print("⚠️ HUGGING FACE TOKEN REQUIRED FOR 24/7 CLOUD HOSTING")
        print("=" * 70)
        print("Aapko ek free Hugging Face Access Token chahiye taaki app cloud par 24/7 live rahe.")
        print("1. Kholein: https://huggingface.co/settings/tokens")
        print("2. 'Create new token' par click karein aur Role: 'Write' chunein.")
        print("3. Phir ye command chalayein:")
        print(f"   python deploy_to_hf_space.py --token <your_hf_token>")
        print("=" * 70 + "\n")
        return None

    api = HfApi(token=hf_token)
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"\n[✓] Authenticated as Hugging Face user: @{username}")
    except Exception as e:
        print(f"\n[✗] Authentication failed with token: {e}")
        return None

    repo_id = f"{username}/{space_name}"
    print(f"[*] Creating / Syncing Hugging Face Space: {repo_id} (Gradio SDK)...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            token=hf_token,
            exist_ok=True
        )
        print(f"[✓] Space confirmed: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"[!] Note: {e}")

    # Set up git remote for Hugging Face Space
    remote_url = f"https://{username}:{hf_token}@huggingface.co/spaces/{repo_id}"
    print("[*] Pushing files to Hugging Face Space git repository...")
    
    # Check if hf remote already exists
    remotes = subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.splitlines()
    if "hf-space" in remotes:
        subprocess.run(["git", "remote", "remove", "hf-space"], capture_output=True)
    
    subprocess.run(["git", "remote", "add", "hf-space", remote_url], check=True)
    push_res = subprocess.run(["git", "push", "-f", "hf-space", "main"], capture_output=True, text=True)
    
    if push_res.returncode == 0:
        permanent_url = f"https://huggingface.co/spaces/{repo_id}"
        embed_url = f"https://{username}-{space_name.replace('_', '-')}.hf.space"
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! 24/7 CLOUD SPACE IS NOW LIVE!")
        print("=" * 70)
        print(f"• Main Space URL: {permanent_url}")
        print(f"• Direct Fullscreen App URL: {embed_url}")
        print("\nYeh link aapke PC ke band hone ke baad bhi 24/7 mobile aur doosre system par kaam karega!")
        print("=" * 70 + "\n")
        return permanent_url
    else:
        print(f"[!] Push warning: {push_res.stderr}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--token", help="Hugging Face User Access Token (Write permissions)")
    parser.add_argument("--name", default="dramaclaw-vox-studio", help="Space name")
    args = parser.parse_args()
    deploy(token=args.token, space_name=args.name)
