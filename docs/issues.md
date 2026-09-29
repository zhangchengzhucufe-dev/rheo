# 跨会话问题日志

发现**别人板块**的问题（坑、接口不对、规格缺失）：记在这里，不要顺手实现。
自己板块的 bug 直接修。

| 日期 | 会话 | 发现 | 影响 | 建议 |
|---|---|---|---|---|
| 2026-09-29 | B | PR #2（`feat/analysis`）分支基点过老：diff 含整个仓库骨架，且带 `rheotrace/__init__.py` 的占位版改动 | 与已合入 main 的 rheotrace 包（#3）直接冲突，无法干净 merge | merge 前 rebase 到最新 main；`rheotrace/*` 以 main 为准（占位 docstring 已被真实现取代），删掉本分支对该文件的改动 |
| 2026-09-29 | B→C | 互校 PR #2（canon.py/adapters.py）与已冻结的 rheotrace spec：接口语义无损（ver_bump 可从 weight_sync 派生、committed 可从 segment_end 派生、n_workers↔world_size），但**文件序≠t0 序**——rheotrace 事件按信封 ts（=区间结束 t_end）排序，不同 lane 的 span 区间大量重叠，而 canon 要求 t0 非降序（乱序即拒） | adapters.py 的 spec→canon reader 若逐行流式转换会全部被拒 | adapter 里先 `rheotrace.read()` 整读、转换后按 t0 重排再喂 canon（M1 规模整读无压力）；`.gz` 直接用 `rheotrace.iread` 透明解压，勿用 `path.read_bytes()` 自行嗅探（gzip 二进制会猜错格式） |
| | | | | |
