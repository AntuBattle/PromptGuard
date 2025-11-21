#!/usr/bin/env python3
"""
Utility script to pull Ollama models from the CLI.

Example:
    python scripts/load_ollama_model.py llama3 mistral
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict


def ensure_ollama_available() -> None:
    if shutil.which("ollama") is None:
        raise SystemExit(
            "The 'ollama' CLI was not found on PATH. "
            "Install Ollama from https://ollama.com/ and ensure the CLI is accessible."
        )


def model_is_present(model: str, env: Dict[str, str]) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:  # pragma: no cover - handled by ensure_ollama_available
        raise SystemExit(f"ollama CLI is not available: {exc}") from exc
    return result.returncode == 0


def pull_model(model: str, env: Dict[str, str]) -> None:
    print(f"[PromptGuard] Pulling Ollama model '{model}'...")
    subprocess.run(["ollama", "pull", model], check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull or refresh local Ollama models.")
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more Ollama model identifiers (e.g., 'llama3', 'mistral:7b').",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("OLLAMA_HOST"),
        help="Custom Ollama host URL (overrides OLLAMA_HOST).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Always pull, even if the model already exists locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_ollama_available()

    env = os.environ.copy()
    if args.host:
        env["OLLAMA_HOST"] = args.host

    for model in args.models:
        if not args.force and model_is_present(model, env):
            print(f"[PromptGuard] Model '{model}' already available locally. Skipping pull.")
            continue
        try:
            pull_model(model, env)
        except subprocess.CalledProcessError as exc:
            print(f"[PromptGuard] Failed to pull '{model}': {exc}", file=sys.stderr)
            return exc.returncode or 1

    print("[PromptGuard] Requested models are ready for use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
