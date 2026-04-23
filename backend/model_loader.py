"""
model_loader.py

Loads the CodeT5 (or fine-tuned) model once at startup and exposes a
thread-safe generate() function used by the FastAPI route.

Supports:
  - Salesforce/codet5p-770m-py  (decoder-only, AutoModelForCausalLM)
  - Salesforce/codet5-base       (encoder-decoder, AutoModelForSeq2SeqLM)
  - Local fine-tuned checkpoint  (set MODEL_PATH env var)
  - PEFT/LoRA adapter checkpoints
"""

import logging
import threading
import time
from pathlib import Path

import torch
from cachetools import LRUCache
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

import config

logger = logging.getLogger("model_loader")

# ---------------------------------------------------------------------------
# Globals (loaded once)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_lock = threading.Lock()
_cache: LRUCache = LRUCache(maxsize=config.CACHE_SIZE)
_device = "cpu"
_is_causal = False  # True for decoder-only models like codet5p-770m-py

# Models that use causal (decoder-only) architecture
_CAUSAL_MODEL_IDS = {
    "salesforce/codet5p-2b",
    "salesforce/codet5p-6b",
    "salesforce/codet5p-16b",
}


def _is_causal_model(model_path: str) -> bool:
    """Detect whether the model is decoder-only based on its name."""
    return model_path.lower().rstrip("/") in _CAUSAL_MODEL_IDS


def _load():
    global _model, _tokenizer, _device, _is_causal

    model_path = config.MODEL_PATH
    logger.info(f"Loading model from: {model_path}")

    start = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Ensure pad token exists (needed for batched generation)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    local_path = Path(model_path)

    # ── PEFT adapter ────────────────────────────────────────────────────────
    if local_path.exists() and (local_path / "adapter_config.json").exists():
        logger.info("Detected PEFT adapter — loading with PEFT...")
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_id = peft_config.base_model_name_or_path
        _is_causal = _is_causal_model(base_id)

        if _is_causal:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_id, torch_dtype=torch.float32, trust_remote_code=True
            )
        else:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_id, torch_dtype=torch.float32
            )
        _model = PeftModel.from_pretrained(base_model, model_path)

    # ── Standard model ──────────────────────────────────────────────────────
    else:
        _is_causal = _is_causal_model(model_path)
        if _is_causal:
            logger.info("Loading as causal (decoder-only) model...")
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        else:
            logger.info("Loading as seq2seq (encoder-decoder) model...")
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )

    _model.eval()

    if torch.cuda.is_available():
        _device = "cuda"
        _model = _model.to(_device)
        logger.info("Running on CUDA GPU")
    else:
        _device = "cpu"
        logger.info("Running on CPU")

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s  (causal={_is_causal})")


def load_model():
    """Call once at startup."""
    with _lock:
        if _model is None:
            _load()


def generate(code_context: str) -> str:
    """
    Generate a code completion for the given context.
    Results are cached by context string (LRU).
    """
    cached = _cache.get(code_context)
    if cached is not None:
        return cached

    with _lock:
        inputs = _tokenizer(
            code_context,
            return_tensors="pt",
            max_length=config.MAX_INPUT_TOKENS,
            truncation=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        gen_kwargs: dict = dict(
            max_new_tokens=config.MAX_NEW_TOKENS,
            num_beams=config.NUM_BEAMS,
            early_stopping=True,
            pad_token_id=_tokenizer.pad_token_id,
        )
        if config.DO_SAMPLE:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = config.TEMPERATURE

        with torch.no_grad():
            output_ids = _model.generate(**inputs, **gen_kwargs)

        if _is_causal:
            # For causal models, strip the input tokens from the output
            input_len = inputs["input_ids"].shape[1]
            suggestion = _tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
        else:
            suggestion = _tokenizer.decode(output_ids[0], skip_special_tokens=True)

    _cache[code_context] = suggestion
    return suggestion
