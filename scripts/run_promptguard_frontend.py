#!/usr/bin/env python3
"""
Start a lightweight local web UI to exercise PromptGuard filters.

The frontend exposes a simple form where you can type a prompt, pick the filter
implementation (NLP, ML, or LLM), and immediately see whether the prompt would
be blocked along with the structured score returned by the filter.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs

from promptguard._log import configure_logger
from promptguard.filtered_output import FilteredOutput
from promptguard.llm_filter import LLMFilter
from promptguard.ml_filter import MLFilter
from promptguard.nlp_filter import NLPFilter


def format_score(score: object) -> str:
    try:
        return json.dumps(score, indent=2, ensure_ascii=False)
    except TypeError:
        return str(score)


def is_blocked(result: FilteredOutput) -> bool:
    score = result.score
    label = None
    if isinstance(score, dict):
        label = score.get("label")
    if label is not None:
        try:
            return int(label) == 1
        except (TypeError, ValueError):
            pass
    eval_flag = getattr(result, "eval", None)
    if isinstance(eval_flag, str):
        return eval_flag.upper() in {"BLOCKED", "UNSAFE"}
    return False


class FilterManager:
    def __init__(
        self,
        llm_provider: str,
        llm_model: Optional[str],
        llm_base_url: Optional[str],
    ) -> None:
        self._instances: dict[str, object] = {}
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url

    def _get_filter(self, name: str):
        name = name.lower()
        if name not in self._instances:
            if name == "llm":
                self._instances[name] = LLMFilter(
                    provider=self._llm_provider,
                    model=self._llm_model,
                    base_url=self._llm_base_url,
                )
            elif name == "ml":
                self._instances[name] = MLFilter()
            elif name == "nlp":
                self._instances[name] = NLPFilter()
            else:
                raise ValueError(f"Unknown filter '{name}'.")
        return self._instances[name]

    def evaluate(self, name: str, prompt: str) -> FilteredOutput:
        filt = self._get_filter(name)
        return filt.safe_eval(prompt)  # type: ignore[no-any-return]


class PromptGuardHandler(BaseHTTPRequestHandler):
    filter_manager: FilterManager

    def do_GET(self):  # noqa: N802 - HTTP verb hook
        self._render_page()

    def do_POST(self):  # noqa: N802 - HTTP verb hook
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = parse_qs(body)

        prompt = data.get("prompt", [""])[0].strip()
        filter_choice = data.get("filter", ["llm"])[0]

        error: Optional[str] = None
        evaluation = None

        if not prompt:
            error = "Please enter a prompt to evaluate."
        else:
            try:
                result = self.filter_manager.evaluate(filter_choice, prompt)
                evaluation = {
                    "blocked": is_blocked(result),
                    "output": result.output,
                    "score": result.score,
                    "filter": filter_choice.lower(),
                }
            except Exception as exc:  # noqa: BLE001 - surface errors to UI
                error = f"Filter execution failed: {exc}"

        self._render_page(prompt, filter_choice, evaluation, error)

    def log_message(self, format: str, *args):  # noqa: A003 - match BaseHTTPRequestHandler signature
        # Reduce noise by routing server logs through standard print
        print(f"[Frontend] {self.address_string()} - {format % args}")

    def _render_page(
        self,
        prompt: str = "",
        filter_choice: str = "llm",
        evaluation: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        status_html = ""
        if error:
            status_html = f'<div class="status error">{html.escape(error)}</div>'
        elif evaluation:
            status = "Blocked" if evaluation["blocked"] else "Allowed"
            css = "blocked" if evaluation["blocked"] else "safe"
            score_pre = html.escape(format_score(evaluation["score"]))
            output = html.escape(evaluation["output"])
            status_html = (
                f'<div class="status {css}">'
                f"<strong>{status} by {evaluation['filter'].upper()} filter.</strong>"
                f"<p>Sanitized output:</p><pre>{output}</pre>"
                f"<p>Score payload:</p><pre>{score_pre}</pre>"
                "</div>"
            )

        selected = {
            "llm": "selected" if filter_choice.lower() == "llm" else "",
            "ml": "selected" if filter_choice.lower() == "ml" else "",
            "nlp": "selected" if filter_choice.lower() == "nlp" else "",
        }

        escaped_prompt = html.escape(prompt)
        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>PromptGuard Playground</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; background: #f4f6f8; color: #111; }}
        h1 {{ margin-bottom: 0.5rem; }}
        form {{ background: #fff; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
        textarea {{ width: 100%; min-height: 120px; margin: 0.5rem 0; padding: 0.75rem; font-size: 1rem; }}
        select, button {{ padding: 0.5rem; font-size: 1rem; }}
        .status {{ margin-top: 1rem; padding: 1rem; border-radius: 6px; background: #fff; border: 1px solid #e0e0e0; }}
        .status.safe {{ border-color: #2e7d32; }}
        .status.blocked {{ border-color: #c62828; }}
        .status.error {{ border-color: #ef6c00; }}
        pre {{ background: #f0f0f0; padding: 0.75rem; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>PromptGuard Playground</h1>
    <p>Test prompts against NLP, ML, or LLM filters without leaving your browser.</p>
    <form method="POST">
        <label for="filter">Filter</label>
        <select name="filter" id="filter">
            <option value="llm" {selected['llm']}>LLM Filter</option>
            <option value="ml" {selected['ml']}>ML Filter</option>
            <option value="nlp" {selected['nlp']}>NLP Filter</option>
        </select>
        <label for="prompt">Prompt</label>
        <textarea name="prompt" id="prompt" required placeholder="Enter a prompt to evaluate...">{escaped_prompt}</textarea>
        <button type="submit">Run Filter</button>
    </form>
    {status_html}
</body>
</html>"""
        payload = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the PromptGuard web playground.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765).")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=os.environ.get("PG_LLM_PROVIDER", "openai_compatible"),
        help="LLMFilter provider (openai_compatible, anthropic, ollama).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("PG_LLM_MODEL"),
        help="LLM model identifier (falls back to filter defaults).",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=os.environ.get("PG_LLM_BASE_URL"),
        help="Optional base URL for OpenAI-compatible backends or Ollama host.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("PG_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for server logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logger(args.log_level)

    handler_cls = PromptGuardHandler
    handler_cls.filter_manager = FilterManager(args.llm_provider, args.llm_model, args.llm_base_url)

    server = HTTPServer((args.host, args.port), handler_cls)
    print(f"[PromptGuard] Web frontend available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PromptGuard] Shutting down frontend...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
