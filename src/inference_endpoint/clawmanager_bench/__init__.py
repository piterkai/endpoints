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

"""ClawManager tool-execution load test.

Inverts the Edge-Agentic benchmark (see ``load_generator/agentic_inference_strategy.py``):
there, a real LLM endpoint is driven with canned/replayed tool results. Here, the LLM
is mocked with scripted responses (see ``mock_llm_server.py``) while a ClawManager
deployment (a separate system under test, not part of this repository) provisions
real agent instances that actually execute the scripted tool calls, so the CPU cost
of real tool execution can be measured at N-way concurrency.

Not part of the HTTP-endpoint benchmarking pipeline: none of ``HTTPEndpointClient``,
the ZMQ transport, or the metrics_aggregator subprocess are used here. This package
talks to ClawManager's REST/WebSocket API and to its own in-process mock LLM server.
See ``examples/12_ClawManager_ToolExec_Benchmark/README.md`` for usage.
"""
