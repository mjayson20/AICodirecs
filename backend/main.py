"""
main.py — FastAPI inference server for AI Code Suggester.

Endpoints:
  POST /complete   — Generate code completion
  GET  /health     — Health check
  GET  /stats      — Usage statistics

Run:
    python main.py
    # or
    uvicorn main:app --host 127.0.0.1 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import model_loader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Stats (in-memory, resets on restart)
# ---------------------------------------------------------------------------

_stats = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
    "cache_hits": 0,
}


# ---------------------------------------------------------------------------
# Lifespan — load model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Code Suggester backend...")
    model_loader.load_model()
    logger.info(f"Server ready. Model: {config.MODEL_PATH}")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Code Suggester",
    description="Local AI-powered code completion API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # VS Code extension on localhost
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CompleteRequest(BaseModel):
    code_context: str = Field(
        ...,
        description="The code snippet preceding the cursor",
        min_length=1,
        max_length=4096,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Override max tokens for this request",
        ge=1,
        le=256,
    )


class CompleteResponse(BaseModel):
    suggestion: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str


class StatsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    errors: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=config.MODEL_PATH)


@app.get("/stats", response_model=StatsResponse)
async def stats():
    total = _stats["total_requests"]
    avg = (_stats["total_latency_ms"] / total) if total > 0 else 0.0
    return StatsResponse(
        total_requests=total,
        avg_latency_ms=round(avg, 2),
        errors=_stats["errors"],
    )


@app.post("/complete", response_model=CompleteResponse)
async def complete(req: CompleteRequest, request: Request):
    _stats["total_requests"] += 1

    # Temporarily override max tokens if requested
    original_max = config.MAX_NEW_TOKENS
    if req.max_tokens:
        config.MAX_NEW_TOKENS = req.max_tokens

    t0 = time.perf_counter()
    try:
        suggestion = model_loader.generate(req.code_context)
    except Exception as exc:
        _stats["errors"] += 1
        logger.error(f"Generation error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        config.MAX_NEW_TOKENS = original_max

    latency_ms = (time.perf_counter() - t0) * 1000
    _stats["total_latency_ms"] += latency_ms

    logger.info(
        f"[/complete] latency={latency_ms:.0f}ms  "
        f"input_len={len(req.code_context)}  "
        f"suggestion_len={len(suggestion)}"
    )

    return CompleteResponse(suggestion=suggestion, latency_ms=round(latency_ms, 2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
