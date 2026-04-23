"""
collect_github_data.py

Collects Python functions from public GitHub repositories using the
GitHub Search API (unauthenticated = 60 req/hr, authenticated = 5000 req/hr).

Usage:
    python collect_github_data.py --output ../model/data/raw.jsonl --limit 20000
    python collect_github_data.py --output ../model/data/raw.jsonl --limit 20000 --token YOUR_GITHUB_TOKEN
"""

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

GITHUB_SEARCH_URL = "https://api.github.com/search/code"
GITHUB_RAW_URL = "https://raw.githubusercontent.com"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "ai-code-suggester-dataset-collector",
}

# Popular Python repos to seed the search (avoids rate-limit heavy search API)
SEED_REPOS = [
    "psf/requests",
    "pallets/flask",
    "django/django",
    "numpy/numpy",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "pytorch/pytorch",
    "keras-team/keras",
    "fastapi/fastapi",
    "tiangolo/fastapi",
    "sqlalchemy/sqlalchemy",
    "aio-libs/aiohttp",
    "encode/httpx",
    "pydantic/pydantic",
    "python/cpython",
    "home-assistant/core",
    "ansible/ansible",
    "scrapy/scrapy",
    "celery/celery",
    "boto/boto3",
    "aws/aws-cli",
    "docker/compose",
    "pytest-dev/pytest",
    "pypa/pip",
    "python-poetry/poetry",
]


def set_token(token: str):
    if token:
        HEADERS["Authorization"] = f"token {token}"


def get_repo_python_files(repo: str, session: requests.Session) -> list[str]:
    """Return a list of Python file paths in a repo (up to 100)."""
    url = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        return [
            item["path"]
            for item in tree
            if item["type"] == "blob" and item["path"].endswith(".py")
        ][:100]  # cap per repo
    except Exception:
        return []


def fetch_file_content(repo: str, path: str, session: requests.Session) -> str | None:
    """Fetch raw file content from GitHub."""
    url = f"{GITHUB_RAW_URL}/{repo}/HEAD/{path}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# AST-based function extractor
# ---------------------------------------------------------------------------

def extract_functions(source: str) -> list[dict]:
    """
    Parse Python source and extract top-level and class-level functions.
    Returns list of dicts with keys: name, source, docstring.
    """
    results = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return results

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        try:
            func_lines = source.splitlines()[node.lineno - 1 : node.end_lineno]
            func_source = "\n".join(func_lines).strip()

            # Skip very short or very long functions
            if len(func_lines) < 3 or len(func_lines) > 80:
                continue

            # Skip test functions
            if node.name.startswith("test_"):
                continue

            docstring = ast.get_docstring(node) or ""

            results.append(
                {
                    "name": node.name,
                    "source": func_source,
                    "docstring": docstring,
                    "num_lines": len(func_lines),
                }
            )
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(output_path: str, limit: int, token: str | None):
    set_token(token or os.environ.get("GITHUB_TOKEN", ""))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    collected = 0
    seen_hashes: set[int] = set()

    with open(output_path, "w", encoding="utf-8") as out_f:
        pbar = tqdm(total=limit, desc="Collecting functions")

        for repo in SEED_REPOS:
            if collected >= limit:
                break

            tqdm.write(f"  → {repo}")
            py_files = get_repo_python_files(repo, session)

            for file_path in py_files:
                if collected >= limit:
                    break

                content = fetch_file_content(repo, file_path, session)
                if not content:
                    continue

                functions = extract_functions(content)
                for fn in functions:
                    if collected >= limit:
                        break

                    # Deduplicate by source hash
                    src_hash = hash(fn["source"])
                    if src_hash in seen_hashes:
                        continue
                    seen_hashes.add(src_hash)

                    record = {
                        "repo": repo,
                        "file": file_path,
                        "name": fn["name"],
                        "source": fn["source"],
                        "docstring": fn["docstring"],
                        "num_lines": fn["num_lines"],
                    }
                    out_f.write(json.dumps(record) + "\n")
                    collected += 1
                    pbar.update(1)

                # Respect rate limits
                time.sleep(0.1)

        pbar.close()

    print(f"\nDone. Collected {collected} functions → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Python functions from GitHub")
    parser.add_argument("--output", default="../model/data/raw.jsonl")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--token", default=None, help="GitHub personal access token")
    args = parser.parse_args()

    collect(args.output, args.limit, args.token)
