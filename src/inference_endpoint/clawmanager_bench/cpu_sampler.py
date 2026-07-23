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

"""CPU sampling for instances under load.

ClawManager itself exposes no Kubernetes metrics-server/cAdvisor integration —
``GET /api/v1/instances/:id/runtime`` returns only the latest self-reported
``system_info`` snapshot (an untyped map on the server side; schema not
guaranteed across runtime versions), refreshed roughly every 5s by the in-pod
agent. ``RuntimePollCPUSampler`` polls this repeatedly to build a time series.
``KubectlTopCPUSampler`` is an optional supplement for users with direct
cluster + metrics-server access, giving real per-pod CPU millicores instead of
self-reported load averages.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .clawmanager_client import ClawManagerClient

_KUBECTL_TOP_LINE_RE = re.compile(
    r"^(?P<name>\S+)\s+(?P<cpu>\d+)m\s+(?P<memory>\d+)Mi\s*$"
)


@dataclass(frozen=True)
class CPUSample:
    instance_id: int
    timestamp: float
    cpu_cores: float | None
    load_1m: float | None
    load_5m: float | None
    load_15m: float | None


@dataclass
class CPUSampleSeries:
    samples: list[CPUSample] = field(default_factory=list)

    def for_instance(self, instance_id: int) -> list[CPUSample]:
        return [s for s in self.samples if s.instance_id == instance_id]

    def cpu_seconds(self, instance_id: int) -> float:
        """Approximate CPU-seconds via trapezoidal integration of ``load_1m *
        cpu_cores`` over the sampled timestamps.

        This is an approximation of a self-reported load average, not a
        measurement of actual consumed CPU-seconds — treat it as a rough
        proxy, not an exact figure.
        """
        points = [
            (s.timestamp, s.load_1m * s.cpu_cores)
            for s in self.for_instance(instance_id)
            if s.load_1m is not None and s.cpu_cores is not None
        ]
        if len(points) < 2:
            return 0.0
        total = 0.0
        # Pairwise adjacent-point iteration: points[1:] is deliberately one
        # shorter than points, so strict= does not apply here.
        for (t0, v0), (t1, v1) in zip(points, points[1:]):  # noqa: B905
            total += (t1 - t0) * (v0 + v1) / 2.0
        return total


class CPUSampler(Protocol):
    async def start(self, instance_ids: list[int]) -> None: ...
    async def stop(self) -> CPUSampleSeries: ...


def extract_cpu_metrics(
    system_info: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Defensively extract ``cpu.{cores,load.{1m,5m,15m}}`` from a ``system_info``
    payload of unknown/unversioned shape. Returns ``(cores, load_1m, load_5m, load_15m)``,
    with ``None`` for any field that is missing or not a number.
    """

    def _num(value: Any) -> float | None:
        return value if isinstance(value, (int, float)) else None

    if not isinstance(system_info, dict):
        return None, None, None, None
    cpu = system_info.get("cpu")
    if not isinstance(cpu, dict):
        return None, None, None, None
    load = cpu.get("load")
    load = load if isinstance(load, dict) else {}
    return (
        _num(cpu.get("cores")),
        _num(load.get("1m")),
        _num(load.get("5m")),
        _num(load.get("15m")),
    )


class RuntimePollCPUSampler:
    """Polls ``GET /instances/:id/runtime`` on an interval via ``ClawManagerClient``."""

    def __init__(
        self, client: ClawManagerClient, *, poll_interval_s: float = 5.0
    ) -> None:
        self._client = client
        self._poll_interval_s = poll_interval_s
        self._series = CPUSampleSeries()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def _poll_loop(self, instance_ids: list[int]) -> None:
        while not self._stop_event.is_set():
            for instance_id in instance_ids:
                try:
                    details = await self._client.get_runtime_details(instance_id)
                except Exception:  # noqa: BLE001 — one instance's poll failure must not stop the rest
                    continue
                system_info = (details.get("runtime") or {}).get("system_info")
                cores, load_1m, load_5m, load_15m = extract_cpu_metrics(system_info)
                self._series.samples.append(
                    CPUSample(
                        instance_id=instance_id,
                        timestamp=time.time(),
                        cpu_cores=cores,
                        load_1m=load_1m,
                        load_5m=load_5m,
                        load_15m=load_15m,
                    )
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval_s
                )
            except TimeoutError:
                pass

    async def start(self, instance_ids: list[int]) -> None:
        self._stop_event.clear()
        self._task = asyncio.ensure_future(self._poll_loop(instance_ids))

    async def stop(self) -> CPUSampleSeries:
        self._stop_event.set()
        if self._task is not None:
            await self._task
        return self._series


def parse_kubectl_top_output(text: str) -> dict[str, float]:
    """Parses ``kubectl top pod --no-headers`` output into ``{pod_name: cpu_millicores}``."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        match = _KUBECTL_TOP_LINE_RE.match(line.strip())
        if match:
            result[match.group("name")] = float(match.group("cpu"))
    return result


class KubectlTopCPUSampler:
    """Shells out to ``kubectl top pod`` on an interval.

    Requires ``kubectl`` on PATH, cluster access, and metrics-server installed
    (none of which ClawManager itself provides) — a pod-name-to-instance-id
    mapping must be supplied since ClawManager's pod naming is deployment-specific.
    """

    def __init__(
        self,
        namespace: str,
        pod_name_by_instance_id: dict[int, str],
        *,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._namespace = namespace
        self._pod_name_by_instance_id = pod_name_by_instance_id
        self._poll_interval_s = poll_interval_s
        self._series = CPUSampleSeries()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def _run_kubectl_top(self) -> str:
        proc = subprocess.run(
            ["kubectl", "top", "pod", "-n", self._namespace, "--no-headers"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout

    async def _poll_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                output = await loop.run_in_executor(None, self._run_kubectl_top)
                millicores_by_pod = parse_kubectl_top_output(output)
            except (subprocess.CalledProcessError, FileNotFoundError):
                millicores_by_pod = {}
            now = time.time()
            for instance_id, pod_name in self._pod_name_by_instance_id.items():
                millicores = millicores_by_pod.get(pod_name)
                # cpu_seconds() integrates load_1m * cpu_cores; setting load_1m=1.0
                # here makes that integral equal actual measured cores-in-use over
                # time, i.e. real CPU-seconds rather than a load-average proxy.
                self._series.samples.append(
                    CPUSample(
                        instance_id=instance_id,
                        timestamp=now,
                        cpu_cores=(millicores / 1000.0)
                        if millicores is not None
                        else None,
                        load_1m=1.0 if millicores is not None else None,
                        load_5m=None,
                        load_15m=None,
                    )
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval_s
                )
            except TimeoutError:
                pass

    async def start(self, instance_ids: list[int]) -> None:
        self._stop_event.clear()
        self._task = asyncio.ensure_future(self._poll_loop())

    async def stop(self) -> CPUSampleSeries:
        self._stop_event.set()
        if self._task is not None:
            await self._task
        return self._series
