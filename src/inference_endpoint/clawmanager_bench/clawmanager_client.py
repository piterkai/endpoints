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

"""Thin REST client for the ClawManager control-plane API.

Covers only what this benchmark needs (login, model registration, instance
CRUD, runtime/CPU polling) — not the full ClawManager API surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .exceptions import ClawManagerAPIError


@dataclass(frozen=True)
class CreateInstanceRequest:
    """Mirrors ClawManager's ``CreateInstanceRequest`` (instance_handler.go) field-for-field."""

    name: str
    type: str
    cpu_cores: float
    memory_gb: int
    disk_gb: int
    os_type: str
    os_version: str
    description: str | None = None
    mode: str | None = None
    gpu_enabled: bool = False
    gpu_count: int = 0
    image_registry: str | None = None
    image_tag: str | None = None
    environment_overrides: dict[str, str] = field(default_factory=dict)
    storage_class: str = ""

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_gb": self.disk_gb,
            "os_type": self.os_type,
            "os_version": self.os_version,
            "gpu_enabled": self.gpu_enabled,
            "gpu_count": self.gpu_count,
            "storage_class": self.storage_class,
        }
        if self.description is not None:
            body["description"] = self.description
        if self.mode is not None:
            body["mode"] = self.mode
        if self.image_registry is not None:
            body["image_registry"] = self.image_registry
        if self.image_tag is not None:
            body["image_tag"] = self.image_tag
        if self.environment_overrides:
            body["environment_overrides"] = self.environment_overrides
        return body


class ClawManagerClient:
    """Authenticated REST client bound to one ClawManager deployment."""

    def __init__(
        self, base_url: str, session: aiohttp.ClientSession, access_token: str
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session
        self.access_token = access_token

    @classmethod
    async def login(
        cls,
        base_url: str,
        username: str,
        password: str,
        session: aiohttp.ClientSession,
    ) -> ClawManagerClient:
        """``POST /api/v1/auth/login`` — obtains a bearer JWT for an admin-role account."""
        url = f"{base_url.rstrip('/')}/api/v1/auth/login"
        async with session.post(
            url, json={"username": username, "password": password}
        ) as response:
            body = await response.text()
            if response.status != 200:
                raise ClawManagerAPIError(response.status, body, url)
            payload = await response.json()
        return cls(base_url, session, payload["access_token"])

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with self._session.request(
            method, url, headers=self._headers(), **kwargs
        ) as response:
            body = await response.text()
            if response.status >= 400:
                raise ClawManagerAPIError(response.status, body, url)
            if not body:
                return {}
            return await response.json()

    async def upsert_llm_model(
        self,
        *,
        display_name: str,
        base_url: str,
        provider_model_name: str,
        provider_type: str = "openai",
        protocol_type: str = "openai",
        is_active: bool = True,
    ) -> dict[str, Any]:
        """``PUT /api/v1/admin/models`` — registers (or updates) an LLM model row.

        Callers are responsible for ensuring this is the only *active* model (or
        the only one ``CLAWMANAGER_LLM_MODEL=auto`` would resolve to) before
        creating instances — this client does not deactivate other rows.
        """
        return await self._request(
            "PUT",
            "/api/v1/admin/models",
            json={
                "display_name": display_name,
                "base_url": base_url,
                "provider_model_name": provider_model_name,
                "provider_type": provider_type,
                "protocol_type": protocol_type,
                "is_active": is_active,
                "is_secure": False,
            },
        )

    async def create_instance(self, req: CreateInstanceRequest) -> dict[str, Any]:
        """``POST /api/v1/instances``."""
        return await self._request("POST", "/api/v1/instances", json=req.to_json())

    async def start_instance(self, instance_id: int) -> dict[str, Any]:
        """``POST /api/v1/instances/:id/start``."""
        return await self._request("POST", f"/api/v1/instances/{instance_id}/start")

    async def get_runtime_details(self, instance_id: int) -> dict[str, Any]:
        """``GET /api/v1/instances/:id/runtime``.

        Returns the latest self-reported snapshot only (not a time series) —
        poll repeatedly to build one; see ``cpu_sampler.py``.
        """
        return await self._request("GET", f"/api/v1/instances/{instance_id}/runtime")

    async def delete_instance(self, instance_id: int) -> dict[str, Any]:
        """``DELETE /api/v1/instances/:id``."""
        return await self._request("DELETE", f"/api/v1/instances/{instance_id}")
