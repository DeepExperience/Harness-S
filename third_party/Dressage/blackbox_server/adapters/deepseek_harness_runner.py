"""One-shot DeepSeek Harness launcher used by the blackbox adapter.

The adapter owns the Dressage proxy and process lifecycle.  This helper only
loads the selected frozen composition, runs one task, and emits one JSON result
on stdout so backend logs cannot be mistaken for the final assistant message.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from blackbox_server.adapters.dsh_termination import budget_stop_reason, classify_cli_termination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness-id",
        choices=(
            "minimal",
            "standard-core-native",
            "standard",
            "task-environment-native",
            "dressage-claw-native",
        ),
        required=True,
    )
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--minimal-composition", type=Path)
    parser.add_argument("--model-overlay", type=Path)
    parser.add_argument("--core-patch", type=Path)
    parser.add_argument("--environment-patch", type=Path)
    parser.add_argument("--harness-overlay", type=Path)
    parser.add_argument("--dsh-executable")
    parser.add_argument("--jsonrpc-executable")
    return parser.parse_args()


def _jsonl_metrics(session_root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    terminal_reasons: list[Any] = []
    malformed = 0
    paths = sorted(session_root.rglob("*.jsonl")) + sorted(
        session_root.rglob("*.jsonl.zstd")
    )
    for path in paths:
        with ExitStack() as stack:
            if path.name.endswith(".zstd"):
                import zstandard

                handle = stack.enter_context(
                    zstandard.open(path, mode="rt", encoding="utf-8", errors="replace")
                )
            else:
                handle = stack.enter_context(
                    path.open(encoding="utf-8", errors="replace")
                )
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                event_type = str(event.get("type") or "unknown")
                counts[event_type] += 1
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                for key in (
                    "inputTokens",
                    "outputTokens",
                    "reasoningTokens",
                    "cacheReadTokens",
                    "cacheWriteTokens",
                ):
                    value = raw_usage.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        usage[key] += int(value)
                if event_type == "turn/end":
                    terminal_reasons.append(data.get("reason"))
    return {
        "event_counts": dict(sorted(counts.items())),
        "usage": dict(usage),
        "tool_calls": counts.get("tool/call", 0),
        "tool_results": counts.get("tool/result", 0),
        "malformed_jsonl_lines": malformed,
        "terminal_reasons": terminal_reasons,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _harness_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.harness_id == "minimal":
        layers = [args.minimal_composition]
    else:
        layers = [args.model_overlay]
        if args.harness_id in {
            "standard-core-native",
            "task-environment-native",
            "dressage-claw-native",
        }:
            layers.append(args.core_patch)
        if args.harness_id in {"task-environment-native", "dressage-claw-native"}:
            layers.append(args.environment_patch)
            if args.harness_overlay is not None:
                layers.append(args.harness_overlay)
    return {
        "harness_id": args.harness_id,
        "layers": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in layers
            if path is not None
        ],
    }


def _stream_tail(value: bytes | str | None, limit: int = 4000) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value or ""
    return text.strip()[-limit:]


def _format_process_error(exc: subprocess.CalledProcessError) -> str:
    """Preserve both DSH streams; the CLI sometimes reports failures on stdout."""

    stdout = _stream_tail(exc.stdout if exc.stdout is not None else exc.output)
    stderr = _stream_tail(exc.stderr)
    return (
        f"DeepSeek Harness exited with code {exc.returncode}; "
        f"stdout_tail={stdout!r}; stderr_tail={stderr!r}"
    )


def _run_minimal(args: argparse.Namespace, task: str) -> str:
    if args.minimal_composition is None or args.jsonrpc_executable is None:
        raise ValueError(
            "minimal requires --minimal-composition and --jsonrpc-executable"
        )
    from deepseek_harness import DeepSeekHarness

    with DeepSeekHarness(
        provider="deepseek-official",
        model=args.model,
        max_tokens=args.max_tokens,
        cwd=str(args.workspace),
        session_root=str(args.session_root),
        cordis=str(args.minimal_composition),
        runtime_bin=args.jsonrpc_executable,
    ) as harness:
        result = harness.run(task, session_id=args.session_id)
    return result.final_response or ""


async def _run_cli(args: argparse.Namespace, task: str) -> tuple[str, str]:
    if args.dsh_executable is None or args.model_overlay is None:
        raise ValueError("standard harnesses require --dsh-executable and --model-overlay")
    # Task repositories may pin an older Node in their Conda environment.
    # Pin only the controller interpreter; preserve PATH for task tools.
    node = os.environ.get("HAPT_DSH_NODE_EXECUTABLE")
    command = ([node] if node else []) + [
        args.dsh_executable,
        "--profile",
        "headless",
        "--patch",
        str(args.model_overlay),
    ]
    if args.harness_id in {
        "standard-core-native",
        "task-environment-native",
        "dressage-claw-native",
    }:
        if args.core_patch is None:
            raise ValueError(f"{args.harness_id} requires --core-patch")
        command.extend(["--patch", str(args.core_patch)])
    if args.harness_id in {"task-environment-native", "dressage-claw-native"}:
        if args.environment_patch is None:
            raise ValueError(
                f"{args.harness_id} requires --environment-patch"
            )
        command.extend(["--patch", str(args.environment_patch)])
        if args.harness_overlay is not None:
            if not args.harness_overlay.is_file():
                raise ValueError(
                    f"task Harness overlay is missing: {args.harness_overlay}"
                )
            command.extend(["--patch", str(args.harness_overlay)])
    command.append(task)
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(args.workspace),
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        await process.wait()
        raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return (
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
    )


async def async_main() -> int:
    args = parse_args()
    task = args.task_file.read_text(encoding="utf-8")
    args.session_root.mkdir(parents=True, exist_ok=True)
    stderr = ""
    if args.harness_id == "minimal":
        final_response = await asyncio.to_thread(_run_minimal, args, task)
    else:
        try:
            final_response, stderr = await _run_cli(args, task)
        except subprocess.CalledProcessError as exc:
            metrics = _jsonl_metrics(args.session_root.parent)
            decision = classify_cli_termination(exc.returncode, metrics)
            termination = decision["kind"] if decision["score_patch"] else None
            diagnostic = {"returncode": exc.returncode, "metrics": metrics,
                          "termination": termination or "unknown_process_exit",
                          "stdout": (exc.output or b"").decode("utf-8", errors="replace"),
                          "stderr": (exc.stderr or b"").decode("utf-8", errors="replace")}
            (args.session_root / "runner-failure.json").write_text(json.dumps(diagnostic, ensure_ascii=False))
            # Unknown process errors remain infrastructure failures. Budget exits
            # retain their proxy trajectory and are scored by the environment.
            if termination is None or metrics["malformed_jsonl_lines"]:
                # Preserve the actual CLI stderr in the HTTP backend error,
                # before container cleanup removes runner-failure.json.
                import sys
                print(diagnostic["stderr"][-4000:], file=sys.stderr, flush=True)
                raise
            print(json.dumps({"final_response": diagnostic["stdout"], "stderr": diagnostic["stderr"],
                              "metrics": metrics, "harness_config": _harness_config(args),
                              "termination": termination, "original_returncode": exc.returncode}, ensure_ascii=False))
            return 0
    print(
        json.dumps(
            {
                "final_response": final_response,
                "stderr": stderr,
                "metrics": _jsonl_metrics(args.session_root.parent),
                "harness_config": _harness_config(args),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except subprocess.CalledProcessError as exc:
        print(_format_process_error(exc), file=os.sys.stderr)
        return int(exc.returncode or 1)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
