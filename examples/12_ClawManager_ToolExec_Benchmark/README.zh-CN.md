# ClawManager 工具执行压测（Tool-Execution Benchmark）

> [English version](README.md)

本工具与 [Edge-Agentic benchmark](../11_Edge_Agentic_Example/README.md)（另见
`docs/load_generator/DESIGN.md` 中的 "AgenticInferenceStrategy" 一节）的思路相反：
Edge-Agentic 是把一段固定的对话轨迹回放给**真实的 LLM endpoint**，而 tool 调用的
结果只是数据集里预先录制好的静态内容，从未真正执行过任何 tool。这里则是反过来：
**LLM 被替换为模拟（mock）**（返回确定性的脚本化 `tool_calls`），而由一个真实的
ClawManager 部署（一个独立的被测系统，不属于本仓库）创建真正的 agent 实例，
**真正执行**这些脚本化的 tool 调用。目的是在 N 路并发下，测量真实 tool 执行
（shell/进程执行、文件读写/搜索、代码执行）在被测系统上的 CPU 开销，而不是
LLM 推理本身的开销。

代码实现位于 `src/inference_endpoint/clawmanager_bench/`。它**不属于**基于 HTTP
endpoint 的压测主链路——ClawManager 本身不是一个 LLM HTTP endpoint，因此这里
完全不使用 `HTTPEndpointClient`、ZMQ 传输层，也不使用 metrics_aggregator 子进程。

## 工作原理

1. 本地启动一个 mock LLM server（`mock_llm_server.py`），并将其注册为你的
   ClawManager 部署上的一个激活模型（`PUT /api/v1/admin/models`）——此后创建的
   每一个 ClawManager 实例都会把 LLM 流量路由到这个 mock server。
2. 通过 ClawManager 的 REST API 创建并启动 N 个 agent 实例。
3. 每个实例都通过其交互式 shell WebSocket（`GET /api/v1/instances/:id/shell`）
   来驱动——ClawManager 没有无头（headless）的"启动任务"接口，所以要启动一段
   对话，只能往一个实时终端里输入 CLI 命令（见下方 **Step 0**）。
4. 实例自身的 agent 进程会调用（被 mock 的）LLM，收到脚本化的 `tool_calls`，
   并**真正在 pod 内部执行它们**——shell 命令、文件 I/O、代码执行。本压测
   测量的正是这部分真实执行的 CPU 开销。
5. 真实的 tool 执行结果会自动被带入下一次 LLM 调用（标准的 agent 循环行为）；
   mock server 通过"对话 id 标记 + 已完成的轮次计数"而非内容本身，来匹配每个
   请求应返回的脚本化回复——因为真实 tool 的输出永远不会与任何预录制的参考值
   完全一致。
6. 通过轮询 ClawManager 的单实例 runtime 接口来采样 CPU（如果你有直接的集群
   访问权限，也可以选用 `kubectl top pod`）。
7. 最终打印一份报告（已执行的 tool 调用数、吞吐量、各实例的近似 CPU 秒数），
   并可选择写入 JSON 文件。

## 前置条件

- 一个正在运行的 ClawManager 部署，并拥有一个具备管理员权限的账号。
- `uv sync --extra clawmanager-bench`（会添加 `aiohttp`，同时用于 mock LLM
  HTTP server 和 WebSocket shell driver）。

## Step 0：确定你的 CLI 启动模板与轮次标记（turn marker）

有两个配置项**没有安全的默认值**，因为它们取决于你所部署的 runtime 镜像里
实际运行的 OpenClaw/Hermes CLI 是什么样子——这部分对本仓库来说是外部黑盒。
在运行压测之前：

1. 手动创建一个实例（通过 ClawManager 的 UI 或 API），并打开它的 shell
   （`GET /api/v1/instances/:id/shell`，例如用某个 WebSocket 命令行工具，或者
   ClawManager 网页 UI 自带的终端）。
2. 输入用于启动一段 agent 对话的命令，并记下其精确语法——这将成为
   `cli_launch_template`，其中 `{conversation_id}` 和 `{message}` 两个占位符
   会在运行时被替换填入。
3. 观察终端在两轮对话之间会打印什么（例如 shell 提示符重新出现、或某个特定
   的 banner），并写一个只匹配这个特征的正则表达式——这将成为
   `turn_marker_regex`。默认自带的值（`\$\s*$`，一个裸的 shell 提示符）
   对于真实的 agent CLI 几乎肯定是不对的。

## 运行

本目录下的 `config.yaml` 列出了完整的字段及其含义；目前还没有实现 YAML 加载器，
所以需要把它们作为 CLI flag 传入：

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

`--confirm-only-active-model` 必须显式传入：注册 mock LLM 模型会影响该部署上
所有解析 `model=auto` 的其他实例。

`trajectories/` 目录下提供了两份对话轨迹数据集：

- `shell_and_file_ops.jsonl` —— 覆盖 `shell_exec`、`file_write`、`file_read`、`file_search`。
- `code_exec.jsonl` —— 覆盖 `code_exec`，以及一次 `file_read`/`file_write` 往返。

两者都使用与 `AgenticInferenceDataset` 相同的 JSONL schema
（`conversation_id`/`turn`/`role`/`content`/`system`/`tool_calls`/`tool_results`）。
与 Edge-Agentic 不同的是，这里的 `tool_results` 字段仅作参考用途——它们只是
给阅读文件的人展示*预期*结果，并不会被发送给真实的 runtime，因为真实 tool
的输出会由 runtime 自己产生。

## 解读报告

- `throughput_tool_calls_per_s` —— 所有实例上，每秒真实执行的 tool 调用数。
- `cpu_seconds_per_instance` —— 这是一个**近似值**：默认的 `runtime-poll`
  采样器是对 ClawManager 自报的负载均值按时间积分得到的，并不是直接测得的
  CPU 秒数。如果你有直接的集群 + metrics-server 访问权限，还有一个更接近
  真实值的、基于 `kubectl top pod` 的采样器（`cpu_sampler.KubectlTopCPUSampler`），
  但它并未接入 `clawmanager-bench run` 这个 CLI 命令——因为 ClawManager 的 API
  不提供 instance id 到 pod 名称的映射，所以只能直接从 Python 里用你自己的
  映射关系来驱动它（详见该类的 docstring）。
- `per_instance[].failed`/`failure_reason` —— 如果这里出现
  `ShellDriverTimeoutError`，通常意味着 `turn_marker_regex` 没有匹配上你的
  runtime 实际输出的终端内容；请重新检查 Step 0。

## 已知局限

1. `cli_launch_template` 和 `turn_marker_regex` 没有安全的默认值，必须手动
   摸索确定（见 Step 0）。
2. 管理员凭据必须由使用者自行提供；本工具不负责创建账号。
3. `mock_llm_cluster_url` 必须是集群内部可达的地址（DNS/NodePort/Ingress）——
   目前只做了一个简单的 `localhost` 合理性检查。
4. shell WebSocket 的鉴权方式（bearer header 还是 query param）尚未针对真实
   前端验证过；两种方式都已实现，通过 `--shell-auth-mode` 切换，默认使用
   header。
5. 标准的 OpenAI `tool_calls` 报文格式已经在 ClawManager 的 AI Gateway 层
   得到确认，但 runtime 本身是否会原样往返这个格式，在真实跑一次之前无法
   确认——第一次运行时可以加上 `--log-raw-requests-path` 来查看原始请求体。
6. 流式（`"stream": true`）请求默认会以非流式方式应答；精确的 SSE
   `tool_calls` 增量分片尚未实现。
7. 除非你自行补充额外的度量手段，CPU 数值都只是近似值（见上文"解读报告"），
   而非精确的 CPU 秒数。

## 测试

单元测试（`tests/unit/clawmanager_bench/`）覆盖了轨迹加载、对话匹配逻辑、
mock LLM server（对本地实例发起真实 HTTP 请求）、本地参考 tool 执行器、
CPU 指标解析，以及报告聚合逻辑——全部无需连接真实集群即可运行。未来如果
需要针对真实集群的验证，可以在 `tests/integration/clawmanager_bench/` 下
新增一个 `@pytest.mark.integration` 测试，并通过环境变量指向一个真实部署。
