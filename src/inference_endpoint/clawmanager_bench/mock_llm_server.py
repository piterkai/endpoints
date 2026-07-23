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

"""OpenAI-compatible mock LLM server serving scripted, real-tool-triggering replies.

Registered as a ClawManager ``llm_models`` row's ``base_url`` (see
``clawmanager_client.py``), so a real ClawManager-managed agent instance calls
this server instead of a real LLM. Every reply's ``tool_calls`` are scripted, but
the *runtime that receives them executes the tools for real* — this server never
executes a tool itself.

Streaming (``"stream": true``) requests are answered non-streaming by default;
see ``ToolExecMockLLMServer.force_non_streaming``. Exact SSE tool_calls
delta-chunking is the riskiest unverified assumption in this design (see
``examples/12_ClawManager_ToolExec_Benchmark/README.md``) and is deliberately
not implemented until a real run shows it is required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from .conversation_matcher import extract_conversation_id, match_turn
from .trajectory import ScriptedAssistantTurn, ToolExecTrajectorySet

logger = logging.getLogger(__name__)

_TERMINAL_TURN = ScriptedAssistantTurn(
    turn=-1, content="Task complete. No further action needed.", tool_calls=[]
)


@dataclass(frozen=True)
class ServedTurnRecord:
    """One served ``/v1/chat/completions`` response, for report aggregation."""

    conversation_id: str | None
    turn: int
    timestamp: float
    tool_call_count: int


@dataclass
class _RequestLog:
    records: list[ServedTurnRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, record: ServedTurnRecord) -> None:
        with self.lock:
            self.records.append(record)

    def snapshot(self) -> list[ServedTurnRecord]:
        with self.lock:
            return list(self.records)


class ToolExecMockLLMServer:
    """Threaded aiohttp server exposing an OpenAI-compatible chat completions route."""

    def __init__(
        self,
        trajectories: ToolExecTrajectorySet,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        force_non_streaming: bool = True,
        log_raw_requests_path: Path | None = None,
    ) -> None:
        self.trajectories = trajectories
        self.host = host
        self.port = port
        self.force_non_streaming = force_non_streaming
        self.log_raw_requests_path = log_raw_requests_path
        self.request_log = _RequestLog()

        self._actual_port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._port_ready_event = threading.Event()

    @property
    def url(self) -> str:
        port = self._actual_port or self.port
        return f"http://{self.host}:{port}"

    async def _handle_chat_completions(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001 — the runtime under test is opaque; never 500 on a bad body
            return web.json_response(
                {
                    "error": {
                        "message": f"invalid JSON body: {e}",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )

        if self.log_raw_requests_path is not None:
            with self.log_raw_requests_path.open("a") as f:
                f.write(json.dumps(body) + "\n")

        messages: list[dict[str, Any]] = body.get("messages") or []
        conversation_id = extract_conversation_id(messages)
        if conversation_id is None:
            return web.json_response(
                {
                    "error": {
                        "message": (
                            "no conversation marker found in system prompt "
                            "(expected '[cmbench:conv=<id>]')"
                        ),
                        "type": "invalid_request_error",
                        "code": "missing_conversation_marker",
                    }
                },
                status=400,
            )

        turn = match_turn(self.trajectories, conversation_id, messages)
        if turn is None:
            turn = _TERMINAL_TURN

        self.request_log.append(
            ServedTurnRecord(
                conversation_id=conversation_id,
                turn=turn.turn,
                timestamp=time.time(),
                tool_call_count=len(turn.tool_calls),
            )
        )

        message: dict[str, Any] = {"role": "assistant", "content": turn.content}
        if turn.tool_calls:
            message["tool_calls"] = turn.tool_calls

        response_body = {
            "id": f"cmbench-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-llm"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if turn.tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return web.json_response(response_body, status=200)

    async def _handle_list_models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": "mock-llm", "object": "model"}]}
        )

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        app.router.add_get("/v1/models", self._handle_list_models)
        app.router.add_get("/healthz", self._handle_health)

    async def _start_server(self) -> None:
        app = web.Application()
        self._register_routes(app)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        if self.port == 0:
            server_socket = site._server.sockets[0]  # type: ignore[union-attr]
            self._actual_port = server_socket.getsockname()[1]
        else:
            self._actual_port = self.port

        logger.info("ToolExecMockLLMServer started at %s", self.url)
        self._port_ready_event.set()

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            await site.stop()
            await runner.cleanup()

    def _run_server(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())

    def start(self) -> None:
        self._server_thread = threading.Thread(target=self._run_server, daemon=False)
        self._server_thread.start()
        if not self._port_ready_event.wait(timeout=5.0):
            raise RuntimeError("ToolExecMockLLMServer failed to start within timeout")

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._server_thread is not None:
            self._server_thread.join(timeout=5.0)
