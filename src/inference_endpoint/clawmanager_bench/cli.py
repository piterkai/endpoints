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

"""Orchestrator and CLI entry point for the ClawManager tool-execution benchmark.

Every ``ToolExecBenchConfig`` field is exposed as its own CLI flag (cyclopts
flattens the pydantic model, same as ``inference-endpoint probe``):

    inference-endpoint clawmanager-bench run --clawmanager-base-url ... --admin-username ...
    # or, run directly:
    python -m inference_endpoint.clawmanager_bench.cli run --clawmanager-base-url ...

See ``examples/12_ClawManager_ToolExec_Benchmark/README.md`` for a full walkthrough,
including how to discover ``cli_launch_template``/``turn_marker_regex`` for a real
deployment — neither has a safe default.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import replace
from typing import Any

import aiohttp
import cyclopts

from inference_endpoint.async_utils.runner import run_async
from inference_endpoint.exceptions import InputValidationError

from .clawmanager_client import ClawManagerClient, CreateInstanceRequest
from .config import ToolExecBenchConfig
from .conversation_matcher import embed_conversation_marker
from .cpu_sampler import CPUSampler, RuntimePollCPUSampler
from .exceptions import ShellDriverTimeoutError
from .mock_llm_server import ToolExecMockLLMServer
from .report import (
    BenchmarkReport,
    InstanceRunResult,
    aggregate,
    print_summary,
    write_json,
)
from .shell_driver import ShellDriver, build_launch_prompt
from .trajectory import ToolExecTrajectorySet, load_trajectory_set

logger = logging.getLogger(__name__)

clawmanager_bench_app = cyclopts.App(
    name="clawmanager-bench",
    help="Load-test ClawManager tool-call execution with a mocked LLM.",
)


class ClawManagerAPIResponseError(InputValidationError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(
            f"could not find 'id' in create_instance response: {payload!r}"
        )


def _extract_instance_id(payload: dict[str, Any]) -> int:
    """Handles both a flat ``{"id": N, ...}`` and a ``{"data": {"id": N, ...}}``
    envelope, since the exact response shape wasn't verified against a live
    deployment — see README known limitations.
    """
    if "id" in payload:
        return int(payload["id"])
    data = payload.get("data")
    if isinstance(data, dict) and "id" in data:
        return int(data["id"])
    raise ClawManagerAPIResponseError(payload)


async def _drive_one_instance(
    client: ClawManagerClient,
    session: aiohttp.ClientSession,
    instance_id: int,
    conversation_id: str,
    cfg: ToolExecBenchConfig,
    trajectories: ToolExecTrajectorySet,
) -> InstanceRunResult:
    """Opens a shell session, kicks off the scripted conversation, and waits
    for turn boundaries until the trajectory completes or ``duration_s`` elapses.

    ``tool_calls_executed`` is left at 0 here — the mock LLM server's request
    log is the source of truth for that count and is merged in by the caller
    after all instances finish (see ``run_benchmark_async``).
    """
    start = time.time()
    max_turns = len(
        trajectories.assistant_turns_by_conversation.get(conversation_id, [])
    )
    turns_completed = 0
    failed = False
    failure_reason: str | None = None

    driver = ShellDriver(
        cfg.clawmanager_base_url,
        instance_id,
        client.access_token,
        session,
        launch_template=cfg.cli_launch_template,
        turn_marker_regex=cfg.turn_marker_regex,
        io_timeout_s=cfg.shell_io_timeout_s,
        auth_mode=cfg.shell_auth_mode,
    )
    try:
        await driver.connect()
        system_prompt = embed_conversation_marker(
            trajectories.system_prompt_by_conversation.get(conversation_id, ""),
            conversation_id,
        )
        initial_message = trajectories.initial_user_message_by_conversation.get(
            conversation_id, ""
        )
        launch_prompt = build_launch_prompt(
            conversation_id, system_prompt, initial_message
        )
        await driver.send_launch_prompt(conversation_id, launch_prompt)

        deadline = time.time() + cfg.duration_s if cfg.duration_s else None
        while turns_completed < max_turns:
            if deadline is not None and time.time() > deadline:
                break
            await driver.wait_for_turn_boundary()
            turns_completed += 1
    except ShellDriverTimeoutError as e:
        failed = True
        failure_reason = str(e)
    except Exception as e:  # noqa: BLE001 — one instance's failure must not abort the run
        failed = True
        failure_reason = str(e)
    finally:
        await driver.close()

    return InstanceRunResult(
        instance_id=instance_id,
        conversation_id=conversation_id,
        tool_calls_executed=0,
        wall_time_s=time.time() - start,
        turns_completed=turns_completed,
        failed=failed,
        failure_reason=failure_reason,
    )


async def run_benchmark_async(cfg: ToolExecBenchConfig) -> BenchmarkReport:
    trajectories = load_trajectory_set(cfg.trajectory_dataset_path)
    if not trajectories.conversation_ids:
        raise InputValidationError(
            f"trajectory dataset {cfg.trajectory_dataset_path} has no conversations"
        )
    if len(trajectories.conversation_ids) < cfg.num_instances:
        logger.warning(
            "trajectory set has %d conversation(s) but num_instances=%d; "
            "conversations will repeat across instances",
            len(trajectories.conversation_ids),
            cfg.num_instances,
        )

    mock_server = ToolExecMockLLMServer(
        trajectories,
        host=cfg.mock_llm_bind_host,
        port=cfg.mock_llm_bind_port,
        log_raw_requests_path=cfg.log_raw_requests_path,
    )
    mock_server.start()

    try:
        async with aiohttp.ClientSession() as session:
            client = await ClawManagerClient.login(
                cfg.clawmanager_base_url,
                cfg.admin_username,
                cfg.admin_password,
                session,
            )
            await client.upsert_llm_model(
                display_name="clawmanager-bench-mock-llm",
                base_url=f"{cfg.mock_llm_cluster_url.rstrip('/')}/v1",
                provider_model_name="mock-llm",
            )

            instance_ids: list[int] = []
            conversation_id_by_instance: dict[int, str] = {}
            for i in range(cfg.num_instances):
                conversation_id = trajectories.conversation_ids[
                    i % len(trajectories.conversation_ids)
                ]
                created = await client.create_instance(
                    CreateInstanceRequest(
                        name=f"cmbench-{i}-{uuid.uuid4().hex[:8]}",
                        type="openclaw",
                        cpu_cores=1.0,
                        memory_gb=2,
                        disk_gb=10,
                        os_type="linux",
                        os_version="ubuntu-22.04",
                    )
                )
                instance_id = _extract_instance_id(created)
                instance_ids.append(instance_id)
                conversation_id_by_instance[instance_id] = conversation_id

            await asyncio.gather(*(client.start_instance(i) for i in instance_ids))

            sampler: CPUSampler
            if cfg.cpu_sampler == "kubectl":
                # ClawManager's API exposes no instance_id -> pod_name mapping
                # (pod naming is deployment-specific), so this CLI path cannot
                # build KubectlTopCPUSampler's required mapping itself. Use
                # KubectlTopCPUSampler directly from Python with your own
                # mapping instead of this CLI command — see README.
                raise InputValidationError(
                    "cpu_sampler=kubectl is not available via this CLI command: "
                    "ClawManager exposes no API to map instance ids to pod names. "
                    "Use cpu_sampler.KubectlTopCPUSampler directly from Python "
                    "with your own instance_id -> pod_name mapping, or use the "
                    "default cpu_sampler=runtime-poll."
                )
            else:
                sampler = RuntimePollCPUSampler(
                    client, poll_interval_s=cfg.runtime_poll_interval_s
                )
            await sampler.start(instance_ids)

            raw_results = await asyncio.gather(
                *(
                    _drive_one_instance(
                        client,
                        session,
                        iid,
                        conversation_id_by_instance[iid],
                        cfg,
                        trajectories,
                    )
                    for iid in instance_ids
                ),
                return_exceptions=True,
            )
            results: list[InstanceRunResult] = []
            for instance_id, raw in zip(instance_ids, raw_results, strict=True):
                if isinstance(raw, BaseException):
                    results.append(
                        InstanceRunResult(
                            instance_id=instance_id,
                            conversation_id=conversation_id_by_instance[instance_id],
                            tool_calls_executed=0,
                            wall_time_s=0.0,
                            turns_completed=0,
                            failed=True,
                            failure_reason=str(raw),
                        )
                    )
                else:
                    results.append(raw)

            cpu_series = await sampler.stop()

            tool_calls_by_conversation: dict[str, int] = {}
            for record in mock_server.request_log.snapshot():
                if record.conversation_id is not None:
                    tool_calls_by_conversation[record.conversation_id] = (
                        tool_calls_by_conversation.get(record.conversation_id, 0)
                        + record.tool_call_count
                    )
            results = [
                replace(
                    r,
                    tool_calls_executed=tool_calls_by_conversation.get(
                        r.conversation_id, 0
                    ),
                )
                for r in results
            ]

            report = aggregate(results, cpu_series)

            if cfg.teardown_after_run:
                await asyncio.gather(
                    *(client.delete_instance(i) for i in instance_ids),
                    return_exceptions=True,
                )
    finally:
        mock_server.stop()

    if cfg.report_output_path is not None:
        write_json(report, cfg.report_output_path)
    print_summary(report)
    return report


@clawmanager_bench_app.command(name="run")
def run_cmd(*, config: ToolExecBenchConfig) -> None:
    """Run the ClawManager tool-execution load test."""
    run_async(run_benchmark_async(config))


if __name__ == "__main__":
    clawmanager_bench_app()
