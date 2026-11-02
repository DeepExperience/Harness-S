"""Whitebox tool adapter layer."""

from __future__ import annotations

import json
from typing import Any

from dressage.sandbox.provider import SandboxProvider
from dressage.sandbox.types import SandboxLease


class WhiteboxToolAdapter:
    """Map Dressage whitebox tool IDs to sandbox provider capabilities."""

    def __init__(self, provider: SandboxProvider) -> None:
        self._provider = provider

    async def tool_call(
        self,
        lease: SandboxLease,
        tool_id: str,
        tool_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if tool_id == "shell.exec":
            return await self._shell_exec(lease, tool_args)
        if tool_id == "file.read":
            return await self._file_read(lease, tool_args)
        if tool_id == "file.write":
            return await self._file_write(lease, tool_args)
        raise ValueError(f"unsupported whitebox tool_id={tool_id!r}")

    async def _shell_exec(
        self,
        lease: SandboxLease,
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        command = args.get("cmd", args.get("command"))
        if not isinstance(command, (str, list)):
            raise ValueError("shell.exec requires string/list argument 'cmd' or 'command'")
        result = await self._provider.run_command(
            lease,
            command,
            cwd=args.get("cwd"),
            env=args.get("env") if isinstance(args.get("env"), dict) else None,
            timeout=args.get("timeout"),
            stdin=args.get("stdin"),
        )
        metadata = result.to_dict()
        text = result.stdout if result.stdout else result.stderr
        return text, metadata

    async def _file_read(
        self,
        lease: SandboxLease,
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("file.read requires non-empty string argument 'path'")
        encoding = args.get("encoding", "utf-8")
        if encoding is not None and not isinstance(encoding, str):
            raise ValueError("file.read encoding must be a string or null")
        max_bytes = args.get("max_bytes")
        data = await self._provider.read_file(
            lease,
            path,
            encoding=encoding,
            max_bytes=max_bytes if isinstance(max_bytes, int) else None,
        )
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
            return text, {"path": path, "bytes": len(data), "encoding": None}
        return data, {"path": path, "chars": len(data), "encoding": encoding}

    async def _file_write(
        self,
        lease: SandboxLease,
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("file.write requires non-empty string argument 'path'")
        content = args.get("content", "")
        if not isinstance(content, (str, bytes)):
            content = json.dumps(content, ensure_ascii=False)
        encoding = args.get("encoding", "utf-8")
        if encoding is not None and not isinstance(encoding, str):
            raise ValueError("file.write encoding must be a string or null")
        result = await self._provider.write_file(
            lease,
            path,
            content,
            encoding=encoding,
            append=bool(args.get("append", False)),
        )
        return "", {"path": path, **result}
