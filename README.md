# 多粒度图上的智能体节点分类

在文本属性图上，让大模型按需收集证据并完成节点分类。`multi` 先读目标论文和所属超节点。粗图优先是减少细粒度访问的提示策略，模型从首轮起可查询已见的粗细节点或展开已见社区，也可直接提交。粗图保留的是压缩后的邻域信息，分类效果需经实验验证。

## 实验思路

- 离线构建社区粗图、社区摘要和 BGE-M3 嵌入索引。
- 在线使用原生 function calling。`search_1hop` 接收 `queries` 数组，每项传 `center` 和 `k`。`C:` 或 `F:` 决定查询所在层。`expand_community` 同样接收 `queries`，每项传 `community` 和 `k`。`submit` 提交预测类别、证据引用和简短理由。
- 每个子查询的 `k` 是候选总数上限，优先未见候选。同一操作的同一中心每轮只执行一次，中心须在轮开始前已见。按请求顺序执行，各项单独返回状态。模型通过 `neighbors` 或 `members` 查看每个来源返回的有序引用及 `remaining`。`evidence` 按引用保存首次交付的完整正文和结构，`sim` 保留三位小数。相似度以原始目标论文为基准，初始社区也附相似度。
- 每个样本维护追加式多轮对话。工具定义保持固定，类别只在最终提交时填写。
- 对比 `target_only`、`fine_only`、`coarse_only` 和 `multi`，使用同一份固定节点清单。
- SQLite 保存请求、失败记录、轨迹和运行进度，支持失败重试与同实验 ID 断点续跑。评估统计准确率、Macro-F1、Token 用量及缓存命中。

`multi` 和 `coarse_only` 共享目标论文正文、目标节点度数和所属社区摘要。`coarse_only` 只能访问粗图邻接，`fine_only` 只看到细图证据和结构。`target_only` 只使用目标正文。粗图关系的 `weight` 是构图阶段归并的跨社区边数，原图邻接沿用单位权重。

## 文件与运行

| 文件 | 用途 |
| --- | --- |
| `build_coarse_graph.py` | 独立构图阶段 |
| `main.py` | 智能体搜索与推理 |
| `evaluate.py` | 独立评估，可对少量节点评分 |
| `experiment_io.py` | 配置、持久化和模型请求 |
| `scripts/` | 数据准备、环境安装和本地模型服务 |

三个阶段分别启动。

```bash
python build_coarse_graph.py --env experiment.env
python main.py --env experiment.env
python evaluate.py --env experiment.env
```

配置从 `experiment.env.example`、`models.env.example` 和 `remote_api.env.example` 准备。实验参数与模型配置分开，两阶段各自支持本地模型或外部 API。远端模型名、地址和密钥放在 `private/remote_api.env`。Python 主环境使用 3.11，环境依赖见锁文件。

`AGENT_MAX_GRAPH_TOOL_CALLS` 是统一逻辑查询预算，一个原生调用包含三个实际执行的子查询就扣三次。首次执行的空邻域查询也扣一次。`AGENT_MAX_SEARCH_CALLS_PER_ROUND` 限制该轮所有原生图工具调用的子查询总数，超限和超预算项按顺序返回跳过状态。`AGENT_MAX_MODEL_ROUNDS` 包含预留的最终提交轮。最终轮保留完整工具定义并追加简短提交指令。

本地和 API 能力检查均用单个原生调用验证批量参数和工具结果回传。可复用的对话前缀和固定工具列表有利于缓存命中，实际命中以服务商报告为准。网络重试复用同一请求。非法子查询、模式冲突、重复和容量限制使用正常工具反馈。普通文本、无法配对的响应或无法接受的提交可以修复一次。合法分类提交中的不可见引用会过滤并记录。

v3.6 需要新实验 ID。`remaining` 是批次实际交付后该来源仍可返回的候选数，`budget` 是剩余逻辑查询次数，`submit_only` 提示下一轮只提交。逐来源异常附状态，无法按来源分组的错误附从零开始的子查询索引。模型工具消息使用紧凑 JSON 做容量检查，SQLite 轨迹仍保留 `by_source`、未舍入相似度、状态增量及开销，实际发送的 JSON 可在请求历史中重建。

原生调用数 `tool_calls` 来自模型响应，实际执行的子查询数 `logical_queries` 对应原 `graph_tool_calls`。`api_attempts` 统计实际发送次数，`api_retries` 统计同轮同生成序号的额外发送，`repair_attempts` 统计实际发出的协议修复生成，`sample_recoveries` 统计同一样本额外启动的恢复轮次。耗时统计已落盘的样本执行时间，进程中断前未落盘的片段没有可靠耗时。初始证据和检索新增证据分别统计。缺失的 usage 或缓存字段保持未知，离线构图用量和能力探测用量单列。旧实验仍可独立评估。

当前构图阶段依赖另行提供的 Python 3.10 粗化模块，接口说明见 `vendor/README.md`。准备完成的粗图可以直接用于在线推理。

数据集、模型权重、私密配置、原始粗化模块和实验产物保存在运行环境中。仓库维护代码、测试和空白配置模板。
