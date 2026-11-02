"""Trajectory integrity guards used by the stable SWE-Gym recipe."""

from __future__ import annotations

from functools import lru_cache
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

INVALID_TOOL_CALL_FIELD = "swegym_invalid_tool_call"
INVALID_PYTEST_WRITE_FIELD = "swegym_invalid_pytest_write"
INTEGRITY_METADATA_FIELD = "swegym_integrity"
REWARD_OVERRIDE_FIELD = "swegym_integrity_reward_override"

_WRITE_TOOL_NAMES = {
    "edit",
    "write",
    "multiedit",
    "multi_edit",
    "apply_patch",
}
_WEB_TOOL_NAME_PATTERNS = (
    "browser",
    "web_search",
    "websearch",
    "search_query",
    "open_url",
    "fetch_url",
    "webfetch",
    "web_fetch",
)
_PYTEST_CONFIG_BASENAMES = {
    "conftest.py",
    "pytest.ini",
    "tox.ini",
    "setup.cfg",
    "pyproject.toml",
}


def use_swegym_integrity(metadata: dict[str, Any]) -> bool:
    """Return whether the sample uses the SWE-Gym harness reward."""
    return metadata.get("reward_fn") == "swegym_harness_marker"


def reset_integrity_metadata(metadata: dict[str, Any]) -> None:
    """Clear stale integrity fields before a retry or a new session."""
    for key in (
        INVALID_TOOL_CALL_FIELD,
        INVALID_PYTEST_WRITE_FIELD,
        INTEGRITY_METADATA_FIELD,
        REWARD_OVERRIDE_FIELD,
    ):
        metadata.pop(key, None)


def scan_trajectory_integrity(
    trajectory_payload: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Detect task-repository web access and writes to protected test files."""
    violations: list[dict[str, Any]] = []
    repo_slugs = _blocked_repo_slugs(metadata)

    for segment_index, segment in enumerate(trajectory_payload.get("data") or []):
        messages = segment.get("messages") or []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for tool_index, tool_call in enumerate(message.get("tool_calls") or []):
                name, arguments = _tool_call_name_and_args(tool_call)
                location = {
                    "segment_index": segment_index,
                    "message_index": message_index,
                    "tool_index": tool_index,
                    "tool": name,
                }
                if repo_slugs:
                    hit = _blocked_repo_web_hit(name, arguments, repo_slugs)
                    if hit is not None:
                        violations.append({**location, **hit})

                pytest_hit = _pytest_write_hit(name, arguments)
                if pytest_hit is not None:
                    violations.append({**location, **pytest_hit})

    invalid_tool_call = any(
        violation.get("kind") == "invalid_tool_call" for violation in violations
    )
    invalid_pytest_write = any(
        violation.get("kind") == "invalid_pytest_write" for violation in violations
    )
    metadata[INVALID_TOOL_CALL_FIELD] = invalid_tool_call
    metadata[INVALID_PYTEST_WRITE_FIELD] = invalid_pytest_write
    metadata[INTEGRITY_METADATA_FIELD] = {
        "invalid_tool_call": invalid_tool_call,
        "invalid_pytest_write": invalid_pytest_write,
        "violations": violations,
    }
    return metadata[INTEGRITY_METADATA_FIELD]


def has_integrity_violation(metadata: dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    return bool(
        metadata.get(INVALID_TOOL_CALL_FIELD)
        or metadata.get(INVALID_PYTEST_WRITE_FIELD)
    )


def zero_reward_for_integrity_violation(
    metadata: dict[str, Any] | None,
    reward: float,
) -> float:
    """Return zero for a trajectory carrying a stable-run integrity flag."""
    if not has_integrity_violation(metadata):
        return reward
    assert metadata is not None
    metadata[REWARD_OVERRIDE_FIELD] = {
        "original_reward": float(reward),
        "reward": 0.0,
        "reason": "swegym_integrity_violation",
        "invalid_tool_call": bool(metadata.get(INVALID_TOOL_CALL_FIELD)),
        "invalid_pytest_write": bool(metadata.get(INVALID_PYTEST_WRITE_FIELD)),
    }
    return 0.0


def _tool_call_name_and_args(tool_call: Any) -> tuple[str, Any]:
    if not isinstance(tool_call, dict):
        return "", {}
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or tool_call.get("name") or "")
        raw_args = function.get("arguments")
    else:
        name = str(tool_call.get("name") or "")
        raw_args = tool_call.get("arguments")

    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            arguments = {"_raw": raw_args}
    elif isinstance(raw_args, dict):
        arguments = raw_args
    else:
        arguments = {}
    return name, arguments


def _blocked_repo_web_hit(
    tool_name: str,
    arguments: Any,
    repo_slugs: set[str],
) -> dict[str, Any] | None:
    text = _repo_web_access_text(tool_name, arguments)
    if not text:
        return None
    lowered = text.lower()
    for slug in sorted(repo_slugs):
        if any(pattern.search(lowered) for pattern in _repo_url_patterns(slug)):
            return {
                "kind": "invalid_tool_call",
                "reason": "blocked_repo_web_access",
                "repo": slug,
                "snippet": _snippet(text),
            }
    return None


def _repo_web_access_text(tool_name: str, arguments: Any) -> str:
    name = tool_name.lower()
    if name == "bash":
        command = str(arguments.get("command") or arguments.get("cmd") or "")
        fragments = [
            fragment
            for fragment in _shell_command_fragments(command)
            if _shell_fragment_uses_network(fragment)
        ]
        return "\n".join(fragments)
    if any(pattern in name for pattern in _WEB_TOOL_NAME_PATTERNS):
        return _tool_call_text(tool_name, arguments)
    return ""


def _shell_fragment_uses_network(fragment: str) -> bool:
    return any(
        re.search(pattern, fragment, flags=re.IGNORECASE)
        for pattern in (
            r"\b(?:curl|wget)\b",
            r"\bgit\s+(?:clone|fetch|pull|ls-remote|archive)\b",
            r"\bgit\s+remote\s+add\b",
            r"\bgh\s+(?:api|repo|pr|issue)\b",
            r"\bpip(?:3)?\s+install\b.*(?:https?://|git\+https?://)",
            r"\bpython(?:3)?\s+-m\s+pip\s+install\b.*(?:https?://|git\+https?://)",
        )
    )


def _pytest_write_hit(tool_name: str, arguments: Any) -> dict[str, Any] | None:
    name = tool_name.lower()
    if name in _WRITE_TOOL_NAMES:
        path = _argument_path(arguments)
        if _is_pytest_sensitive_path(path):
            return {
                "kind": "invalid_pytest_write",
                "reason": "write_tool_to_pytest_sensitive_path",
                "path": path,
            }

    if name == "bash":
        command = str(arguments.get("command") or arguments.get("cmd") or "")
        path = _pytest_write_path_from_command(command)
        if path:
            return {
                "kind": "invalid_pytest_write",
                "reason": "shell_write_to_pytest_sensitive_path",
                "path": path,
                "snippet": _snippet(command),
            }
    return None


def _tool_call_text(tool_name: str, arguments: Any) -> str:
    try:
        args_text = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except TypeError:
        args_text = str(arguments)
    return f"{tool_name} {args_text}"


def _argument_path(arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return ""
    for key in ("filePath", "file_path", "path", "filepath"):
        value = arguments.get(key)
        if value:
            return str(value)
    return ""


def _blocked_repo_slugs(metadata: dict[str, Any]) -> set[str]:
    slugs = set(_configured_blocked_repo_slugs())
    for key in ("repo", "repository", "github_repo", "repo_full_name"):
        value = metadata.get(key)
        if value:
            slug = _normalize_repo_slug(str(value))
            if slug:
                slugs.add(slug)
    return slugs


@lru_cache(maxsize=8)
def _configured_blocked_repo_slugs() -> tuple[str, ...]:
    slugs: set[str] = set()
    for item in os.environ.get("DRESSAGE_SWEGYM_BLOCKED_REPOS", "").split(","):
        slug = _normalize_repo_slug(item)
        if slug:
            slugs.add(slug)

    path = os.environ.get("DRESSAGE_SWEGYM_BLOCKED_REPOS_FILE")
    if path:
        slugs.update(_repo_slugs_from_file(path))
    return tuple(sorted(slugs))


def _repo_slugs_from_file(path: str) -> set[str]:
    slugs: set[str] = set()
    file_path = Path(path)
    if not file_path.exists():
        return slugs
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                row = text
            if isinstance(row, dict):
                metadata = (
                    row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                )
                candidates = (
                    row.get("repo"),
                    row.get("repository"),
                    metadata.get("repo"),
                    metadata.get("repository"),
                )
            else:
                candidates = (row,)
            for candidate in candidates:
                slug = _normalize_repo_slug(str(candidate or ""))
                if slug:
                    slugs.add(slug)
    return slugs


def _normalize_repo_slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^https?://(?:www\.)?github\.com/", "", text)
    text = text.removesuffix(".git")
    match = re.match(r"^([a-z0-9_.-]+)/([a-z0-9_.-]+)(?:[/#?].*)?$", text)
    if not match:
        return ""
    return f"{match.group(1)}/{match.group(2)}"


@lru_cache(maxsize=1024)
def _repo_url_patterns(slug: str) -> tuple[re.Pattern[str], ...]:
    escaped = re.escape(slug)
    return (
        re.compile(rf"https?://(?:www\.)?github\.com/{escaped}(?:[/?#.\s\"']|$)"),
        re.compile(rf"git@github\.com:{escaped}(?:\.git)?(?:\s|$)"),
        re.compile(
            rf"https?://patch-diff\.githubusercontent\.com/raw/{escaped}"
            rf"(?:[/?#\s\"']|$)"
        ),
        re.compile(rf"https?://raw\.githubusercontent\.com/{escaped}(?:[/?#\s\"']|$)"),
        re.compile(rf"https?://api\.github\.com/repos/{escaped}(?:[/?#\s\"']|$)"),
        re.compile(rf"https?://codeload\.github\.com/{escaped}(?:[/?#\s\"']|$)"),
    )


def _is_pytest_sensitive_path(path: str) -> bool:
    if not path:
        return False
    normalized = path.replace("\\", "/").strip().strip("\"'")
    if not normalized or normalized.startswith("/tmp/"):
        return False
    if "/testbed/" in normalized:
        normalized = normalized.split("/testbed/", 1)[1]
    normalized = normalized.split("::", 1)[0].lstrip("./")
    basename = normalized.rsplit("/", 1)[-1]
    if basename.endswith((".bak", ".orig", "~")):
        return False  # Backup files are not official tests or pytest config.
    parts = normalized.split("/")
    return (
        basename in _PYTEST_CONFIG_BASENAMES
        or "tests" in parts
        or "testing" in parts
        or "r2e_tests" in parts
    )


def _pytest_write_path_from_command(command: str) -> str:
    if not command:
        return ""
    for redirect_path in _redirect_write_paths(command):
        if _is_pytest_sensitive_path(redirect_path):
            return redirect_path
    for path in _shell_write_target_paths(command):
        if _is_pytest_sensitive_path(path):
            return path
    return ""


def _redirect_write_paths(command: str) -> list[str]:
    tokens = _shell_tokens(command)
    paths = [tokens[i + 1][0] for i, (token, operator) in enumerate(tokens[:-1])
             if operator and token in {">", ">>", ">|", "&>", "&>>"}
             and not tokens[i + 1][1]]
    for argv in _shell_argvs(command):
        if argv and argv[0].rsplit("/", 1)[-1] == "tee":
            paths.extend(arg for arg in argv[1:] if not arg.startswith("-"))
    return paths


def _shell_write_target_paths(command: str) -> list[str]:
    paths: list[str] = []
    # Python here-documents are executable code too. Do not regress the prior
    # guard while narrowing Python -c handling to actual Python commands.
    for match in re.finditer(
        r"(?:^|[;&|])\s*python(?:\d+(?:\.\d+)?)?\s+(?:-\s+)?<<-?\s*(['\"]?)(\w+)\1[^\n]*\n(.*?)\n[ \t]*\2(?:\n|$)",
        command, re.MULTILINE | re.DOTALL,
    ):
        paths.extend(_python_write_target_paths(match.group(3)))
    for argv in _shell_argvs(command):
        while argv and (argv[0] in {"command", "sudo", "env"} or re.match(r"^[A-Za-z_]\w*=", argv[0])):
            argv = argv[1:]
        if not argv:
            continue
        name, args = argv[0].rsplit("/", 1)[-1], argv[1:]
        if name in {"bash", "sh"} and len(args) >= 2 and args[0] in {"-c", "-lc"}:
            paths.extend(_redirect_write_paths(args[1]))
            paths.extend(_shell_write_target_paths(args[1]))
            continue
        if name.startswith("python") and "-c" in args:
            i = args.index("-c")
            if i + 1 < len(args):
                paths.extend(_python_write_target_paths(args[i + 1]))
        if name not in {"cp", "mv", "rm", "touch", "truncate", "sed", "perl"}:
            continue
        if name == "sed" and not any(a == "--in-place" or a.startswith("--in-place=") or (a.startswith("-") and not a.startswith("--") and "i" in a) for a in args):
            continue
        if name == "perl" and not any(a.startswith("-") and "i" in a for a in args):
            continue
        operands, destination = [], None
        script_given = False
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--":
                operands.extend(args[i + 1:]); break
            if a in {"-t", "--target-directory"} and name in {"cp", "mv"}:
                destination = args[i + 1] if i + 1 < len(args) else None
                i += 2; continue
            if a.startswith("--target-directory="):
                destination = a.split("=", 1)[1]; i += 1; continue
            valued = {"-e", "--expression", "-f", "--file"} if name in {"sed", "perl"} else {"-r", "--reference", "-d", "--date", "-s", "--size"} if name in {"touch", "truncate"} else set()
            if a in valued:
                script_given = True
                i += 2; continue
            if name in {"sed", "perl"} and (a.startswith("-e") or a.startswith("--expression=")):
                script_given = True
            if a.startswith("-"):
                i += 1; continue
            operands.append(a); i += 1
        if name in {"sed", "perl"} and not script_given:
            operands = operands[1:]
        if name == "cp":
            paths.extend([destination] if destination else operands[-1:])
        elif name == "mv":
            paths.extend(operands + ([destination] if destination else []))
        else:
            paths.extend(operands)
    return paths


def _shell_tokens(command: str) -> list[tuple[str, bool]]:
    """Preserve quoting so prose and Python strings never become shell operators."""
    try:
        lexer = shlex.shlex(command, posix=False, punctuation_chars=";&|()<>\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = []
        for raw in lexer:
            quoted = raw.startswith(("'", '"'))
            token = shlex.split(raw)[0] if quoted else raw
            tokens.append((token, not quoted and bool(token) and all(c in ";&|()<>\n" for c in token)))
        return tokens
    except ValueError:
        return []  # Incomplete tool arguments are checked by the tool parser.


def _shell_argvs(command: str) -> list[list[str]]:
    """Return command operands with redirections removed, without executing code."""
    tokens = _shell_tokens(command)
    groups, current = [], []
    i = 0
    while i < len(tokens):
        token, operator = tokens[i]
        if operator and ("<" in token or ">" in token):
            if current and current[-1].isdigit():
                current.pop()  # File descriptor, e.g. 2>/dev/null.
            i += 2  # Redirection target is not a cp/mv operand.
            continue
        if operator:
            if current:
                groups.append(current); current = []
        else:
            current.append(token)
        i += 1
    if current:
        groups.append(current)
    return groups


def _shell_command_fragments(command: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"\s*(?:&&|\|\||;|\n|\|)\s*", command)
        if part.strip()
    ]


def _python_write_target_paths(command: str) -> list[str]:
    paths = [
        match.group(1)
        for match in re.finditer(
            r"\bopen\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa+]",
            command,
        )
    ]
    for pattern in (
        r"\b(?:unlink|remove|rename|replace)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\bshutil\.(?:move|rmtree)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\bshutil\.(?:copy|copy2|copyfile|move)\s*\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]([^'\"]+)['\"]",
        r"['\"]([^'\"]+)['\"]\s*\)\.write_(?:text|bytes)\s*\(",
    ):
        paths.extend(match.group(1) for match in re.finditer(pattern, command))
    return paths


def _candidate_paths(text: str) -> list[str]:
    paths = [
        match.group(1) for match in re.finditer(r"(/testbed/[^'\"\s;&|()<>]+)", text)
    ]
    paths.extend(
        match.group(1)
        for match in re.finditer(
            r"((?:tests|testing|r2e_tests)/[^'\"\s;&|()<>]+|"
            r"[^'\"\s;&|()<>]*test[^'\"\s;&|()<>]*\.py|"
            r"conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|pyproject\.toml)",
            text,
        )
    )
    return paths


def _snippet(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
