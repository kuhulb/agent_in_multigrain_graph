# 多粒度图上的智能体节点分类

在文本属性图上，让大模型按需收集证据并完成节点分类。先读目标论文和所属超节点，证据充分时直接提交，需要补充时优先查粗图，再按缺口查看原图。智能体可以在粗细图之间继续取证，用逐次 1-hop 查询控制邻域规模。

## 实验思路

- 离线构建社区粗图、社区摘要和 BGE-M3 嵌入索引。
- 在线使用原生 function calling。`search_1hop(center, k)` 按 `C:` 或 `F:` 引用查询相邻社区或论文，`expand_community(community, k)` 读取社区成员，`submit` 提交分类。
- 每次工具调用只有一个中心，查询可以同轮批量发出。按与目标论文的相似度选取候选，首次返回完整文本，重复证据用已读占位。
- 每个样本维护追加式多轮对话。工具定义保持固定，类别只在最终提交时填写。
- 对比 `target_only`、`fine_only`、`coarse_only` 和 `multi`，使用同一份固定节点清单。
- SQLite 保存请求、失败记录、轨迹和运行进度，支持失败重试与同实验 ID 断点续跑。评估统计准确率、Macro-F1、Token 用量及缓存命中。

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

当前构图阶段依赖另行提供的 Python 3.10 粗化模块，接口说明见 `vendor/README.md`。准备完成的粗图可以直接用于在线推理。

数据集、模型权重、私密配置、原始粗化模块和实验产物保存在运行环境中。仓库维护代码、测试和空白配置模板。
