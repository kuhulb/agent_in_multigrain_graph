# 粗化模块接口

当前离线构图需要另行提供 lv_V5.cpython-310.pyc，放在此目录，或通过 GRAPH_LOUVAIN_BYTECODE 指定路径。该依赖由 Python 3.10 子进程加载，主实验环境使用 Python 3.11。

scripts/louvain_worker.py 提供模块调用边界。依赖缺失时构图会明确报错。使用已有粗图进行推理和评估时无需加载此模块。
