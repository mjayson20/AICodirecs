# AI Code Suggester

A fully local, free, open-source AI-powered code autocomplete system — similar to GitHub Copilot — built with CodeT5, FastAPI, and a VS Code extension.

## Architecture

```
User types in VS Code
       ↓
VS Code Extension (TypeScript)
       ↓  POST /complete
FastAPI Backend (Python)
       ↓
CodeT5 / CodeLlama (local inference via Hugging Face / llama.cpp)
       ↓
Inline ghost-text suggestion
```

## Project Structure

```
ai-code-suggester/
├── backend/          # FastAPI inference server
├── model/            # Training scripts and dataset utilities
├── extension/        # VS Code extension (TypeScript)
├── scripts/          # Data collection and preprocessing
├── notebooks/        # Google Colab training notebook
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- 8 GB RAM minimum (16 GB recommended)
- ~5 GB disk space for model weights

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Inference latency | < 1s (CPU) |
| Dataset size | 10K–20K samples |
| Max context tokens | 256 |
| Max completion tokens | 128 |

---

## Free Resources Used

| Resource | Purpose |
|----------|---------|
| Google Colab (free tier) | QLoRA fine-tuning on T4 GPU |
| Hugging Face Hub (free) | Model weights download |
| GitHub REST API (free) | Dataset collection |
| Local CPU | Inference |

## Model Options

| Model | Size | RAM Required | Quality |
|-------|------|-------------|---------|
| `Salesforce/codet5-base` | 900 MB | 4 GB | Good (recommended) |
| `Salesforce/codet5p-220m` | 220 MB | 2 GB | Fast, lighter |
| `codellama/CodeLlama-7b-hf` | 13 GB (4-bit: ~4 GB) | 8 GB | Best quality |

Default is `codet5-base` for CPU compatibility. Switch to CodeLlama after fine-tuning on Colab.
