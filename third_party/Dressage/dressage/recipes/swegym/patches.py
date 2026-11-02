"""One patch boundary for training and standalone SWE-Gym evaluation."""
from pathlib import PurePosixPath
import codecs
import re
import shlex


def extract_patch_command(workdir: str, base_commit: str = "") -> str:
    command = f"cd {shlex.quote(workdir)} && git add -A && git diff --cached --binary --submodule=diff"
    if base_commit:
        command += f" {shlex.quote(base_commit)} --"
    return command


def is_protected_path(path: str) -> bool:
    parsed = PurePosixPath(path.replace("\\", "/").strip("/"))
    parts = tuple(p.lower() for p in parsed.parts)
    name = parsed.name.lower()
    return bool(any(p in {"test", "tests", "testing", "test-data", "r2e_tests", ".git"} for p in parts)
                or name.startswith("test_") or name.endswith("_test.py")
                or name in {"conftest.py", "pytest.ini", "tox.ini"})


def _paths(block: str) -> tuple[str, str]:
    # Git leaves spaces unquoted, but quotes non-ASCII paths using C octal
    # escapes. Prefer unambiguous extended headers over shell tokenization.
    def decode(value):
        value = value.rstrip("\t")
        if value.startswith('"') and value.endswith('"'):
            return codecs.escape_decode(value[1:-1].encode())[0].decode("utf-8")
        return value
    lines = block.splitlines()
    if not lines[0].startswith("diff --git "):
        raise ValueError("malformed git patch header")
    headers = []
    for line in lines[1:]:
        if line.startswith(("@@", "GIT binary patch")):
            break
        headers.append(line)
    def header(prefix):
        return next((decode(line[len(prefix):]) for line in headers if line.startswith(prefix)), None)
    old, new = header("rename from ") or header("copy from "), header("rename to ") or header("copy to ")
    if old is None or new is None:
        old, new = header("--- "), header("+++ ")
        if old is not None and old != "/dev/null": old = old.removeprefix("a/")
        if new is not None and new != "/dev/null": new = new.removeprefix("b/")
        if old == "/dev/null": old = new
        if new == "/dev/null": new = old
    if old is None or new is None:
        raw = lines[0][len("diff --git "):]
        fields = re.findall(r'"(?:\\.|[^"\\])*"|\S+', raw)
        if len(fields) == 2:
            old, new = decode(fields[0]).removeprefix("a/"), decode(fields[1]).removeprefix("b/")
        else:
            # Binary/mode-only changes with a space in the unchanged path.
            candidates = [(raw[:m.start()].removeprefix("a/"), raw[m.end():])
                          for m in re.finditer(" b/", raw)]
            same = [(a, b) for a, b in candidates if a == b]
            if len(same) != 1:
                raise ValueError("ambiguous git patch paths")
            old, new = same[0]
    paths = (old, new)
    for path in paths:
        if path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ValueError("unsafe git patch path")
    return paths


def changed_patch_paths(patch: str) -> list[str]:
    return sorted({_paths(block)[1] for block in re.split(r"(?m)(?=^diff --git )", patch) if block.strip()})


def prepare_patch(patch: str) -> tuple[str, list[str], list[str]]:
    """Omit only newly added root-level reproduction scripts from grading.

    They remain in the agent workspace/original patch. Existing tests, config,
    fixtures, symlinks, renames, and files under test directories stay protected.
    Production changes are retained verbatim, including binary hunks.
    """
    if not patch.strip():
        return patch, [], []
    blocks = re.split(r"(?m)(?=^diff --git )", patch)
    kept, ignored, prohibited = [], [], set()
    for block in blocks:
        if not block:
            continue
        old, new = _paths(block)
        name = PurePosixPath(new).name
        lines = block.splitlines()
        new_reproduction = (
            old == new and "/" not in new and "\\" not in new
            and (name.startswith("test_") or name.endswith("_test.py")) and name.endswith(".py")
            and any(line in {"new file mode 100644", "new file mode 100755"} for line in lines)
            and "--- /dev/null" in lines and f"+++ b/{new}" in lines
            and not any(line.startswith(("rename ", "copy ", "deleted file mode")) for line in lines)
        )
        if new_reproduction:
            ignored.append(new)
            continue
        prohibited.update(path for path in (old, new) if is_protected_path(path))
        kept.append(block)
    return "".join(kept), sorted(set(ignored)), sorted(prohibited)
