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

"""Unit tests for ToolExecMockLLMServer, using a real HTTP server (port=0),
mirroring the pattern used by tests/unit/test_http_mock_fixtures.py for
testing/echo_server.py.

A live-cluster @pytest.mark.integration counterpart (posting through a real
ClawManager AI Gateway to a real runtime) could later be added at
tests/integration/clawmanager_bench/ — out of scope here.
"""

from collections.abc import Generator
from pathlib import Path

import aiohttp
import pytest

from inference_endpoint.clawmanager_bench.conversation_matcher import (
    embed_conversation_marker,
)
from inference_endpoint.clawmanager_bench.mock_llm_server import ToolExecMockLLMServer
from inference_endpoint.clawmanager_bench.trajectory import load_trajectory_set


@pytest.fixture
def mock_llm_server(
    trajectory_jsonl_path: Path,
) -> Generator[ToolExecMockLLMServer, None, None]:
    trajectories = load_trajectory_set(trajectory_jsonl_path)
    server = ToolExecMockLLMServer(trajectories, host="127.0.0.1", port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_conversation_marker_returns_400(
    mock_llm_server: ToolExecMockLLMServer,
) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{mock_llm_server.url}/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            assert response.status == 400
            body = await response.json()
            assert body["error"]["code"] == "missing_conversation_marker"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scripted_tool_calls_served_verbatim(
    mock_llm_server: ToolExecMockLLMServer,
) -> None:
    system = embed_conversation_marker("You are a coding agent.", "c1")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{mock_llm_server.url}/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "List files in the repo."},
                ],
            },
        ) as response:
            assert response.status == 200
            body = await response.json()

    message = body["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "shell_exec"
    assert body["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exhausted_trajectory_serves_terminal_response(
    mock_llm_server: ToolExecMockLLMServer,
) -> None:
    system = embed_conversation_marker("You are a coding agent.", "c1")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "List files in the repo."},
        {"role": "tool", "tool_call_id": "call_1", "content": "real output"},
        {"role": "user", "content": "one more thing"},
    ]
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{mock_llm_server.url}/v1/chat/completions",
            json={"model": "mock", "messages": messages},
        ) as response:
            assert response.status == 200
            body = await response.json()

    assert body["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in body["choices"][0]["message"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_request_log_records_served_turns(
    mock_llm_server: ToolExecMockLLMServer,
) -> None:
    system = embed_conversation_marker("You are a coding agent.", "c1")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{mock_llm_server.url}/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "List files in the repo."},
                ],
            },
        ):
            pass

    records = mock_llm_server.request_log.snapshot()
    assert len(records) == 1
    assert records[0].conversation_id == "c1"
    assert records[0].tool_call_count == 1
