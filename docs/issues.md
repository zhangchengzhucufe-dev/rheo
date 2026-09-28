# 跨会话问题日志

发现**别人板块**的问题（坑、接口不对、规格缺失）：记在这里，不要顺手实现。
自己板块的 bug 直接修。

| 日期 | 会话 | 发现 | 影响 | 建议 |
|---|---|---|---|---|
| 2026-09-29 | B | PR #2（`feat/analysis`）分支基点过老：diff 含整个仓库骨架，且带 `rheotrace/__init__.py` 的占位版改动 | 与已合入 main 的 rheotrace 包（#3）直接冲突，无法干净 merge | merge 前 rebase 到最新 main；`rheotrace/*` 以 main 为准（占位 docstring 已被真实现取代），删掉本分支对该文件的改动 |
| 2026-09-29 | B | PR #2（`feat/analysis`）分支基点过老：diff 含整个仓库骨架，且带 `rheotrace/__init__.py` 的占位版改动 | 与已合入 main 的 rheotrace 包（#3）直接冲突，无法干净 merge | merge 前 rebase 到最新 main；`rheotrace/*` 以 main 为准（占位 docstring 已被真实现取代），删掉本分支对该文件的改动 |
| 2026-09-29 | B→C | 互校 PR #2（canon.py/adapters.py）与已冻结的 rheotrace spec：接口语义无损（ver_bump 可从 weight_sync 派生、committed 可从 segment_end 派生、n_workers↔world_size），但**文件序≠t0 序**——rheotrace 事件按信封 ts（=区间结束 t_end）排序，不同 lane 的 span 区间大量重叠，而 canon 要求 t0 非降序（乱序即拒） | adapters.py 的 spec→canon reader 若逐行流式转换会全部被拒 | adapter 里先 `rheotrace.read()` 整读、转换后按 t0 重排再喂 canon（M1 规模整读无压力）；`.gz` 直接用 `rheotrace.iread` 透明解压，勿用 `path.read_bytes()` 自行嗅探（gzip 二进制会猜错格式） |
| 2026-09-29 | C | `rheotrace-spec-v0.md` 尚未出稿，而 C 的指标口径已定（`docs/metrics-v0.md` §6），需要 trace 满足 8 类最小字段需求 | A 插桩、C 分析都在等接口冻结；字段缺失的指标只能砍 | B 定稿 spec 时按 metrics-v0 §6 对账；C 只约束语义，字段命名随 B |
| 2026-09-29 | C | 权重同步必须是**显式区间事件**（t_start/t_end + target_version），只落版本号不够 | 无区间则 P1 weight_sync 停顿无法归类，M2 rollout MFU 恒等式缺一块 | spec 加 sync 区间事件（含传输+应用+翻转全程） |
| 2026-09-29 | C | 执行段需要 `batch_id`（或显式 dispatch 事件） | 算不了批占用率与长尾掉队份额（S1–S3），而这是 partial rollout 立项依据的核心证据 | 段 schema 加 batch_id；不落 dispatch 事件则 spec 写明"派发时刻 = 同批最早段开始"的推导规则 |
| 2026-09-29 | C | 版本时间线：version bump 事件要带时间戳，执行段要带 birth_version | 算不了 stale token 份额（F5，跨版本续跑收益上界） | spec 加 version timeline 事件 |
| 2026-09-29 | C | abort 事件需带 reason + 已生成 token 是否 committed 的标记 | 废 token 率与"含/不含 abort"长度统计口径不定 | abort 事件带 reason 枚举 + committed bool |
| 2026-09-29 | C | 元数据头需含 model（名或 P/L/d）、world_size、并行布局、时钟基准（单调、ns、左闭右开） | 缺了 MFU 与 per-GPU 换算要靠外部参数硬塞，易口径漂移 | spec 的 header 部分固定这些字段；时钟约定一条必须写死 |
| 2026-09-29 | C | 投机解码 accepted/draft token 计数 | v0 不用，但 v1 会要；schema 现在不留字段将来破坏兼容 | spec 留可选字段即可 |
