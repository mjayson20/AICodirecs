"""
config.py — Backend configuration loaded from environment or .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Model ──────────────────────────────────────────────────────────────────
# Set MODEL_PATH to a local fine-tuned checkpoint directory, or leave as
# the Hugging Face model ID to download the base model automatically.
MODEL_PATH: str = os.getenv("MODEL_PATH", "Salesforce/codet5-base")

# ── Inference ──────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS: int  = int(os.getenv("MAX_INPUT_TOKENS", "256"))
MAX_NEW_TOKENS: int    = int(os.getenv("MAX_NEW_TOKENS", "128"))
NUM_BEAMS: int         = int(os.getenv("NUM_BEAMS", "1"))          # 1 = greedy (fastest)
TEMPERATURE: float     = float(os.getenv("TEMPERATURE", "0.2"))
DO_SAMPLE: bool        = os.getenv("DO_SAMPLE", "false").lower() == "true"

# ── Cache ──────────────────────────────────────────────────────────────────
CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "256"))   # LRU cache entries

# ── Server ─────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))
