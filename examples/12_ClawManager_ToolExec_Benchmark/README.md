# ClawManager Tool-Execution Benchmark

> [中文版](README.zh-CN.md)

This inverts the [Edge-Agentic benchmark](../11_Edge_Agentic_Example/README.md) (see
also `docs/load_generator/DESIGN.md`, "AgenticInferenceStrategy"): Edge-Agentic
replays a fixed conversation trajectory against a **real LLM endpoint**, with
tool-call results canned/replayed from the dataset — no tool is ever actually
executed. Here, the **LLM is mocked** (deterministic scripted `tool_calls`)
while a real ClawManager deployment (a separate system under test, not part of
this repository) provisions actual agent instances that **execute the
scripted tool calls for real**. The goal is
to measure the CPU cost of real tool execution (shell/process exec, file
read/write/search, code execution) at N-way concurrency, on the system under
test rather than on the LLM.

Implementation: `src/inference_endpoint/clawmanager_bench/`. Not part of the
HTTP-endpoint benchmarking pipeline — ClawManager isn't an LLM HTTP endpoint,
so none of `HTTPEndpointClient`, the ZMQ transport, or the metrics_aggregator
subprocess are used here.

## How it works

1. A mock LLM server (`mock_llm_server.py`) is started locally and registered
   as an active model on your ClawManager deployment (`PUT /api/v1/admin/models`)
   — every ClawManager instance created afterward routes its LLM traffic to it.
2. N agent instances are created and started via ClawManager's REST API.
3. Each instance is driven through its interactive shell WebSocket
   (`GET /api/v1/instances/:id/shell`) — ClawManager has no headless
   "start a task" API, so kicking off a conversation means typing a CLI
   invocation into a live terminal (see **Step 0** below).
4. The instance's own agent process calls the (mocked) LLM, receives scripted
   `tool_calls`, and **actually executes them** — shell commands, file I/O,
   code execution — inside the pod. That real execution is what this
   benchmark measures the CPU cost of.
5. Real tool results flow back into the next LLM call automatically (standard
   agent-loop behavior); the mock server matches each request to the right
   scripted reply by conversation-id marker + turn count, not by content,
   since real tool output never equals any canned reference value.
6. CPU is sampled by polling ClawManager's per-instance runtime endpoint (or,
   optionally, `kubectl top pod` if you have direct cluster access).
7. A report (tool calls executed, throughput, approximate CPU-seconds per
   instance) is printed and optionally written to JSON.

## Prerequisites

- A running ClawManager deployment with an admin-role account.
- `uv sync --extra clawmanager-bench` (adds `aiohttp`, used for both the mock
  LLM HTTP server and the WebSocket shell driver).

## Step 0: discover your CLI launch template and turn marker

Two config fields have **no safe default** because they depend on whatever
OpenClaw/Hermes CLI your deployed runtime image actually runs — that's
external to this repository. Before running the benchmark:

1. Create one instance by hand (via the ClawManager UI or API) and open its
   shell (`GET /api/v1/instances/:id/shell`, e.g. with a WebSocket CLI tool or
   the ClawManager web UI's own terminal).
2. Type the command that starts an agent conversation, and note its exact
   syntax — this becomes `cli_launch_template`, with `{conversation_id}` and
   `{message}` placeholders substituted in at run time.
3. Watch what the terminal prints between turns (e.g. a shell prompt
   reappearing, a specific banner) and write a regex that matches only that —
   this becomes `turn_marker_regex`. The shipped default (`\$\s*$`, a bare
   shell prompt) is almost certainly wrong for a real agent CLI.

## Running

See `config.yaml` in this directory for the full set of fields and their
meaning; there is no YAML loader yet, so pass them as CLI flags:

```bash
inference-endpoint clawmanager-bench run \
  --clawmanager-base-url https://clawmanager.example.internal \
  --admin-username admin --admin-password "$CLAWMANAGER_ADMIN_PASSWORD" \
  --num-instances 10 \
  --mock-llm-cluster-url http://cmbench-mock.default.svc.cluster.local:8080 \
  --trajectory-dataset-path examples/12_ClawManager_ToolExec_Benchmark/trajectories/shell_and_file_ops.jsonl \
  --cli-launch-template "openclaw chat --conversation {conversation_id} '{message}'" \
  --turn-marker-regex '\$\s*$' \
  --confirm-only-active-model \
  --report-output-path clawmanager_bench_report.json
```

`--confirm-only-active-model` must be passed explicitly: registering the mock
LLM model affects every instance on the deployment that resolves `model=auto`.

Two trajectory datasets are provided (`trajectories/`):

- `shell_and_file_ops.jsonl` — `shell_exec`, `file_write`, `file_read`, `file_search`.
- `code_exec.jsonl` — `code_exec` plus a `file_read`/`file_write` round trip.

Both use the same JSONL schema as `AgenticInferenceDataset`
(`conversation_id`/`turn`/`role`/`content`/`system`/`tool_calls`/`tool_results`).
Unlike Edge-Agentic, the `tool_results` values here are advisory only — they
document the _expected_ result for a human reading the file, but are never
sent to the runtime, since the real runtime supplies real tool output.

## Interpreting the report

- `throughput_tool_calls_per_s` — real tool calls executed per second across
  all instances.
- `cpu_seconds_per_instance` — an **approximation**: with the default
  `runtime-poll` sampler this integrates ClawManager's self-reported load
  average over time, not a direct CPU-seconds measurement. A closer-to-real
  `kubectl top pod`-based sampler (`cpu_sampler.KubectlTopCPUSampler`) exists
  for users with direct cluster + metrics-server access, but is not wired
  into the `clawmanager-bench run` CLI — ClawManager's API exposes no
  instance-id-to-pod-name mapping, so it must be driven from Python directly
  with your own mapping (see the class docstring).
- `per_instance[].failed`/`failure_reason` — a `ShellDriverTimeoutError` here
  usually means `turn_marker_regex` doesn't match your runtime's actual
  terminal output; re-check Step 0.

## Known limitations

1. `cli_launch_template` and `turn_marker_regex` have no safe default and must
   be discovered by hand (Step 0).
2. Admin credentials must be supplied by the operator; this tool does not
   provision an account.
3. `mock_llm_cluster_url` must be reachable from inside the K8s cluster
   (DNS/NodePort/Ingress) — only a cheap `localhost` sanity check is enforced.
4. The shell WebSocket's auth transport (bearer header vs. query param) is
   unverified against a live frontend; both are implemented behind
   `--shell-auth-mode`, defaulting to header.
5. The standard OpenAI `tool_calls` wire shape is confirmed at ClawManager's
   AI Gateway layer, but whether the runtime itself round-trips exactly that
   shape is unverified until a real run — pass `--log-raw-requests-path` on
   the first run to inspect raw request bodies.
6. Streaming (`"stream": true`) requests are answered non-streaming by
   default; exact SSE `tool_calls` delta-chunking is not implemented.
7. CPU numbers are approximations (see "Interpreting the report" above), not
   exact CPU-seconds, unless you supply your own instrumentation.

## Tests

Unit tests (`tests/unit/clawmanager_bench/`) cover trajectory loading, the
conversation-matching logic, the mock LLM server (real HTTP requests against
a local instance), the local reference tool executors, CPU-metric parsing,
and report aggregation — all without a live cluster. A live-cluster
`@pytest.mark.integration` test could later be added under
`tests/integration/clawmanager_bench/`, gated behind an env var pointing at a
real deployment.
