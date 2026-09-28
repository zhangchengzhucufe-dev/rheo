"""L0 Kernel 层（Triton 起步，热点后置 C++/CUDA）。

版本化 paged attention（KV block 带 birth_version/drift_epoch）、投机解码树验证、
融合采样 + token 级 logprob 捕获。M4+ 启动实现，当前为骨架占位（见 PLAN.md §1 L0）。
"""
