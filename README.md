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

## Step 1 — Dataset Collection

```bash
cd scripts
pip install -r requirements.txt
python collect_github_data.py --output ../model/data/raw.jsonl --limit 20000
python preprocess.py --input ../model/data/raw.jsonl --output ../model/data/dataset.jsonl
```

---

## Step 2 — Training (Google Colab — Free T4 GPU)

1. Upload `notebooks/train_qlora.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4)
3. Mount your Google Drive when prompted
4. Run all cells — fine-tuned weights save to `MyDrive/ai-code-suggester/`

Or run locally (slow, CPU only):
```bash
cd model
pip install -r requirements.txt
python train.py --dataset data/dataset.jsonl --output ./checkpoints
```

---

## Step 3 — Backend Server

```bash
cd backend
pip install -r requirements.txt

# Download base model (first run only, ~1.5 GB for CodeT5-base)
python download_model.py

# Start server
python main.py
# Server runs at http://localhost:8000
```

Test it:
```bash
curl -X POST http://localhost:8000/complete \
  -H "Content-Type: application/json" \
  -d '{"code_context": "def calculate_fibonacci(n):\n    "}'
```

---

## Step 4 — VS Code Extension

```bash
cd extension
npm install
npm run compile

# Install extension in VS Code
# Press F5 to open Extension Development Host, OR:
npx vsce package
code --install-extension ai-code-suggester-0.0.1.vsix
```

---

## Usage

1. Start the backend: `cd backend && python main.py`
2. Open VS Code with the extension installed
3. Start typing Python code
4. Ghost-text suggestions appear automatically
5. Press **Tab** to accept, **Escape** to dismiss

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
