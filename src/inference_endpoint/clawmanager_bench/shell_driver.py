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

"""Drives a ClawManager instance's interactive shell WebSocket.

ClawManager exposes no headless "start a task" API — the only way to make an
instance's agent converse is by typing into its live terminal
(``GET /api/v1/instances/:id/shell``, which execs ``tmux``/``bash`` in the pod's
``desktop`` container and streams raw terminal bytes over the WebSocket).

This driver is best-effort. It does not parse a structured protocol — it
scrapes a live terminal. Turn boundaries are detected via a user-supplied regex
against raw (ANSI-stripped) terminal output. Silent hangs, mid-write
reconnects, and prompt-echo false positives are expected failure modes; the
orchestrator (``cli.py``) treats a ``wait_for_turn_boundary`` timeout as a
per-instance failure, not a fatal error, and continues with the remaining
instances.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal

import aiohttp

from .exceptions import ShellDriverTimeoutError

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

ShellAuthMode = Literal["header", "query-param"]


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from raw terminal output."""
    return _ANSI_ESCAPE_RE.sub("", text)


class ShellDriver:
    """WebSocket-backed driver for one instance's interactive shell."""

    def __init__(
        self,
        base_url: str,
        instance_id: int,
        access_token: str,
        session: aiohttp.ClientSession,
        *,
        launch_template: str,
        turn_marker_regex: str,
        io_timeout_s: float = 60.0,
        auth_mode: ShellAuthMode = "header",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.instance_id = instance_id
        self.access_token = access_token
        self._session = session
        self.launch_template = launch_template
        self._marker_re = re.compile(turn_marker_regex)
        self.io_timeout_s = io_timeout_s
        self.auth_mode = auth_mode
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._buffer = ""

    def _ws_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host_and_path = self.base_url.split("://", 1)[1]
        url = f"{scheme}://{host_and_path}/api/v1/instances/{self.instance_id}/shell"
        if self.auth_mode == "query-param":
            url = f"{url}?token={self.access_token}"
        return url

    async def connect(self) -> None:
        headers = (
            {"Authorization": f"Bearer {self.access_token}"}
            if self.auth_mode == "header"
            else None
        )
        self._ws = await self._session.ws_connect(self._ws_url(), headers=headers)

    async def send_launch_prompt(self, conversation_id: str, message: str) -> None:
        """Types the launch template (with placeholders filled) as one input frame.

        ``message`` fills the template's ``{message}`` placeholder — typically
        the combined system prompt + initial user message from
        ``build_launch_prompt``, not just the bare initial user turn.
        """
        assert self._ws is not None, "call connect() first"
        text = self.launch_template.format(
            conversation_id=conversation_id, message=message
        )
        await self._ws.send_json({"type": "input", "data": text})

    async def wait_for_turn_boundary(self) -> str:
        """Accumulate raw terminal output until the turn-marker regex matches.

        Returns the newly-accumulated (ANSI-stripped) text since the last call,
        for transcript logging — not used for correctness (the mock LLM server
        is the source of truth for turn progression).
        """
        assert self._ws is not None, "call connect() first"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.io_timeout_s
        start_len = len(self._buffer)

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ShellDriverTimeoutError(self.instance_id, self.io_timeout_s)
            try:
                msg = await asyncio.wait_for(self._ws.receive(), timeout=remaining)
            except TimeoutError as e:
                raise ShellDriverTimeoutError(
                    self.instance_id, self.io_timeout_s
                ) from e

            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                raise ShellDriverTimeoutError(self.instance_id, self.io_timeout_s)

            chunk = (
                msg.data
                if isinstance(msg.data, str)
                else msg.data.decode(errors="replace")
            )
            self._buffer += strip_ansi(chunk)

            if self._marker_re.search(self._buffer[start_len:]):
                new_text = self._buffer[start_len:]
                return new_text

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()


def build_launch_prompt(
    conversation_id: str, system_prompt: str, initial_message: str
) -> str:
    """Combines the trajectory's system prompt (carrying the conversation marker,
    see ``conversation_matcher.embed_conversation_marker``) and initial user
    message into the single string a ``ShellDriver`` types as terminal input.
    """
    if system_prompt:
        return f"{system_prompt}\n\n{initial_message}"
    return initial_message
