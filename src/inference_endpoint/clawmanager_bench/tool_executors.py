# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A local reference tool executor.

During a real benchmark run, tool calls are executed for real *inside* the
ClawManager-managed agent instance — that opaque runtime's CPU cost is exactly
what this benchmark measures, and this module's code never runs on that path.

``LocalToolExecutor`` exists for two narrower purposes: (1) smoke-testing that a
trajectory dataset's scripted ``tool_calls`` are well-formed and actually
executable (dataset authoring), and (2) an optional local dry run of the
mock-LLM-server / conversation-matcher turn-taking logic without any ClawManager
cluster, by standing in for "the real runtime" in-process.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolExecResult:
    tool_call_id: str
    content: str
    exit_code: int | None = None


class LocalToolExecutor:
    """Executes ``shell_exec``/``file_read``/``file_write``/``file_search``/``code_exec``.

    All file-system tools are confined under ``root`` (resolved and checked with
    ``Path.resolve()`` + ``is_relative_to``) to keep dataset-authoring dry runs from
    touching the developer's filesystem outside a designated sandbox directory.
    """

    def __init__(self, root: Path, *, default_timeout_s: float = 30.0) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.default_timeout_s = default_timeout_s

    def execute(self, tool_call: dict[str, Any]) -> ToolExecResult:
        function = tool_call["function"]
        name = function["name"]
        raw_arguments = function.get("arguments", "{}")
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else dict(raw_arguments)
        )
        handler = {
            "shell_exec": self._shell_exec,
            "file_read": self._file_read,
            "file_write": self._file_write,
            "file_search": self._file_search,
            "code_exec": self._code_exec,
        }.get(name)
        if handler is None:
            raise ValueError(f"Unsupported tool_calls function name: {name!r}")
        return handler(tool_call["id"], arguments)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Path escapes sandbox root: {relative_path!r}")
        return candidate

    def _shell_exec(self, call_id: str, args: dict[str, Any]) -> ToolExecResult:
        command = args["command"]
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        timeout_s = float(args.get("timeout_s", self.default_timeout_s))
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        content = proc.stdout + proc.stderr
        return ToolExecResult(
            tool_call_id=call_id, content=content, exit_code=proc.returncode
        )

    def _code_exec(self, call_id: str, args: dict[str, Any]) -> ToolExecResult:
        code = args["code"]
        timeout_s = float(args.get("timeout_s", self.default_timeout_s))
        proc = subprocess.run(
            [sys.executable, "-c", code],
            shell=False,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        content = proc.stdout + proc.stderr
        return ToolExecResult(
            tool_call_id=call_id, content=content, exit_code=proc.returncode
        )

    def _file_read(self, call_id: str, args: dict[str, Any]) -> ToolExecResult:
        path = self._resolve(args["path"])
        content = path.read_text() if path.exists() else ""
        return ToolExecResult(tool_call_id=call_id, content=content)

    def _file_write(self, call_id: str, args: dict[str, Any]) -> ToolExecResult:
        path = self._resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return ToolExecResult(
            tool_call_id=call_id, content=f"wrote {len(args['content'])} bytes"
        )

    def _file_search(self, call_id: str, args: dict[str, Any]) -> ToolExecResult:
        pattern = args.get("pattern", "*")
        search_root = self._resolve(args.get("root", "."))
        matches = [str(p.relative_to(self.root)) for p in search_root.rglob(pattern)]
        return ToolExecResult(tool_call_id=call_id, content="\n".join(sorted(matches)))
