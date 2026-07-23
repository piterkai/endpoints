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

"""Unit tests for LocalToolExecutor — real subprocesses/files under tmp_path.

This executor is a dataset-authoring/dry-run aid, not the code path exercised
against a real ClawManager cluster (see module docstring in tool_executors.py),
so there is no live-cluster integration counterpart for it.
"""

import json
import sys
from pathlib import Path

import pytest

from inference_endpoint.clawmanager_bench.tool_executors import LocalToolExecutor


def _call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.mark.unit
def test_shell_exec_runs_real_subprocess(tmp_path: Path) -> None:
    executor = LocalToolExecutor(tmp_path)
    result = executor.execute(
        _call("shell_exec", {"command": [sys.executable, "--version"]})
    )
    assert result.exit_code == 0
    assert "Python" in result.content


@pytest.mark.unit
def test_file_write_then_read(tmp_path: Path) -> None:
    executor = LocalToolExecutor(tmp_path)
    executor.execute(
        _call("file_write", {"path": "a.txt", "content": "hello"}, "call_1")
    )
    result = executor.execute(_call("file_read", {"path": "a.txt"}, "call_2"))
    assert result.content == "hello"


@pytest.mark.unit
def test_file_read_missing_file_returns_empty(tmp_path: Path) -> None:
    executor = LocalToolExecutor(tmp_path)
    result = executor.execute(_call("file_read", {"path": "missing.txt"}))
    assert result.content == ""


@pytest.mark.unit
def test_file_search_finds_matching_files(tmp_path: Path) -> None:
    executor = LocalToolExecutor(tmp_path)
    executor.execute(_call("file_write", {"path": "a.txt", "content": "x"}, "call_1"))
    executor.execute(
        _call("file_write", {"path": "sub/b.txt", "content": "y"}, "call_2")
    )
    executor.execute(_call("file_write", {"path": "c.py", "content": "z"}, "call_3"))

    result = executor.execute(_call("file_search", {"root": ".", "pattern": "*.txt"}))
    assert "a.txt" in result.content
    assert "sub/b.txt" in result.content
    assert "c.py" not in result.content


@pytest.mark.unit
def test_file_path_cannot_escape_sandbox_root(tmp_path: Path) -> None:
    executor = LocalToolExecutor(tmp_path)
    with pytest.raises(ValueError, match="escapes sandbox root"):
        executor.execute(_call("file_read", {"path": "../../etc/passwd"}))


@pytest.mark.unit
def test_code_exec_runs_real_python_subprocess(tmp_path: Path) -> None:
    executor = LocalToolExecutor(tmp_path)
    result = executor.execute(_call("code_exec", {"code": "print(1 + 1)"}))
    assert result.exit_code == 0
    assert result.content.strip() == "2"


@pytest.mark.unit
def test_unsupported_tool_name_raises(tmp_path: Path) -> None:
    executor = LocalToolExecutor(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        executor.execute(_call("delete_prod_database", {}))
