"""trace → 报告：MFU 分解、停顿瀑布、长尾分布、tokens/s/GPU。

会话 C 实现；指标口径见 docs/metrics-v0.md（唯一口径来源）。

模块结构：
- canon      规范事件模型 + JSONL 校验解析（内部表示，CANON_FORMAT 以此为唯一来源）
- adapters   格式接缝：rheo-canon-v0 / rheotrace-jsonl → canon（新格式只改这里）
- metrics    全部指标的纯函数实现
- figures    matplotlib 插图
- report     Markdown 报告组装
- __main__   CLI：python -m bench.analysis <trace> [--out DIR] [--peak-tflops X]
"""
