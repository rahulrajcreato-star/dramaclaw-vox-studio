import os
import sys
import json
import itertools
from pathlib import Path
from dotenv import load_dotenv

# Base directory is workspace root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# -----------------------------------------------------------------------------
# 1. Multi-Key Gemini Director Pool (5 Keys)
# -----------------------------------------------------------------------------
GEMINI_KEYS_POOL = [
    ("PRIMARY", os.getenv("GEMINI_API_KEY", "")),
    ("ALPHA_1", os.getenv("GEMINI_API_KEY_ALPHA_1", "")),
    ("ALPHA_2", os.getenv("GEMINI_API_KEY_ALPHA_2", "")),
    ("BETA_1", os.getenv("GEMINI_API_KEY_BETA_1", "")),
    ("BETA_2", os.getenv("GEMINI_API_KEY_BETA_2", "")),
]

VALID_GEMINI_KEYS = [(name, key.strip()) for name, key in GEMINI_KEYS_POOL if key and len(key.strip()) > 10]
if not VALID_GEMINI_KEYS:
    # Fallback to single GEMINI_API_KEY if present
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        VALID_GEMINI_KEYS = [("PRIMARY", k)]

_gemini_cycle = itertools.cycle(VALID_GEMINI_KEYS) if VALID_GEMINI_KEYS else None

def get_next_gemini_key():
    """Returns the next valid Gemini API key in round-robin sequence."""
    if not _gemini_cycle:
        return ("NONE", "")
    return next(_gemini_cycle)

def update_gemini_key(name: str, key_val: str):
    """Dynamically registers or updates a key."""
    global VALID_GEMINI_KEYS, _gemini_cycle
    key_val = key_val.strip()
    existing = dict(VALID_GEMINI_KEYS)
    existing[name] = key_val
    VALID_GEMINI_KEYS = [(n, k) for n, k in existing.items() if k and len(k) > 10]
    if VALID_GEMINI_KEYS:
        _gemini_cycle = itertools.cycle(VALID_GEMINI_KEYS)

# -----------------------------------------------------------------------------
# 2. Cloud & Hugging Face Storage Configuration
# -----------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or ""
HF_DEFAULT_DATASET = os.getenv("HF_STORAGE_REPO", "antigravity-vox-studio/documentary-masters")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# -----------------------------------------------------------------------------
# 3. GridShare Remote GPU & 20GB VRAM Parameters
# -----------------------------------------------------------------------------
DEFAULT_GRIDSHARE_CONFIG = {
    "host": os.getenv("GRIDSHARE_HOST", "relay.gridshare.in"),
    "port": int(os.getenv("GRIDSHARE_PORT", "30000")),
    "user": os.getenv("GRIDSHARE_USER", "root"),
    "ssh_key_path": os.getenv("GRIDSHARE_KEY_PATH", os.path.expanduser("~/.ssh/id_rsa")),
    "comfyui_url": os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
    "api_token": os.getenv("GRIDSHARE_API_TOKEN", "")
}

GPU_VRAM_TARGET_GB = 20.0
MAX_VRAM_ALLOCATION_GB = 18.5  # Safe headroom to prevent CUDA OOM

# -----------------------------------------------------------------------------
# 4. Project Paths
# -----------------------------------------------------------------------------
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "orchestra_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

SHINKAI_REF_IMAGE = STATIC_DIR / "shinkai_roof_ref.jpg"
FENRIR_REF_AUDIO = STATIC_DIR / "gemini_male_fenrir.wav"
F5_CONDITIONED_AUDIO = STATIC_DIR / "f5_tts_gemini_conditioned_sample.wav"
