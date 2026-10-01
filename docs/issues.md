# 跨会话问题日志

发现**别人板块**的问题（坑、接口不对、规格缺失）：记在这里，不要顺手实现。
自己板块的 bug 直接修。

| 日期 | 会话 | 发现 | 影响 | 建议 |
|---|---|---|---|---|
| 2026-09-29 | B | PR #2（`feat/analysis`）分支基点过老：diff 含整个仓库骨架，且带 `rheotrace/__init__.py` 的占位版改动 | 与已合入 main 的 rheotrace 包（#3）直接冲突，无法干净 merge | merge 前 rebase 到最新 main；`rheotrace/*` 以 main 为准（占位 docstring 已被真实现取代），删掉本分支对该文件的改动 |
| 2026-09-29 | B→C | 互校 PR #2（canon.py/adapters.py）与已冻结的 rheotrace spec：接口语义无损（ver_bump 可从 weight_sync 派生、committed 可从 segment_end 派生、n_workers↔world_size），但**文件序≠t0 序**——rheotrace 事件按信封 ts（=区间结束 t_end）排序，不同 lane 的 span 区间大量重叠，而 canon 要求 t0 非降序（乱序即拒） | adapters.py 的 spec→canon reader 若逐行流式转换会全部被拒 | adapter 里先 `rheotrace.read()` 整读、转换后按 t0 重排再喂 canon（M1 规模整读无压力）；`.gz` 直接用 `rheotrace.iread` 透明解压，勿用 `path.read_bytes()` 自行嗅探（gzip 二进制会猜错格式） |
| 2026-09-29 | C | `rheotrace-spec-v0.md` 尚未出稿，而 C 的指标口径已定（`docs/metrics-v0.md` §6），需要 trace 满足 8 类最小字段需求 | A 插桩、C 分析都在等接口冻结；字段缺失的指标只能砍 | ✅ 已解决：spec v0.1 已冻结，逐项对账见 metrics-v0 §6（8 项满足，2 项缺口见下） |
| 2026-09-29 | C | 权重同步必须是**显式区间事件**（t_start/t_end + target_version），只落版本号不够 | 无区间则 P1 weight_sync 停顿无法归类，M2 rollout MFU 恒等式缺一块 | ✅ 已解决：spec §4.2 `weight_sync{t_start,t_end,version}` 即区间事件 |
| 2026-09-29 | C | 执行段需要 `batch_id`（或显式 dispatch 事件） | 算不了批占用率与长尾掉队份额（S1–S3），而这是 partial rollout 立项依据的核心证据 | ✅ **B 已落实**（见下方 B→C 行）：spec v0.1.1 `segment_start` 增可选 `batch_id`，C 适配器已支持读取，A 插桩可填则 S1–S3 脱离近似。过渡期 C 按时间重叠聚类 + W-BATCH-STAGGERED 守卫（流水派发下伪批占用率口径不适用则剔除），见 metrics-v0 §3.3 |
| 2026-09-29 | C | 版本时间线：version bump 事件要带时间戳，执行段要带 birth_version | 算不了 stale token 份额（F5，跨版本续跑收益上界） | ✅ 已解决：spec §4.2/§4.4/§5.1，且 end_version 使 staleness 直方可做 |
| 2026-09-29 | C | abort 事件需带 reason + 已生成 token 是否 committed 的标记 | 废 token 率与"含/不含 abort"长度统计口径不定 | ✅ 基本解决：E15 abort 必带 reason；committed 以"非 aborted"近似。残留歧义：DAPO 组级拒收（组内 finished 但数据被弃）不是 segment abort，C 现无法区分——若 A 侧要精确废 token 率，请 B 考虑可选 `segment_end.trainer_committed` bool |
| 2026-09-29 | C | 元数据头需含 model（名或 P/L/d）、world_size、并行布局、时钟基准（单调、ns、左闭右开） | 缺了 MFU 与 per-GPU 换算要靠外部参数硬塞，易口径漂移 | ✅ 基本解决：run_start 有 model/clock/n_workers/meta；P/L/d 不入 spec，C 走 --model-config |
| 2026-09-29 | C | 投机解码 accepted/draft token 计数 | v0 不用，但 v1 会要；schema 现在不留字段将来破坏兼容 | ✅ 已解决：spec §8 明确 M5 以新增事件类型做兼容增量 |
| 2026-09-29 | C | 区间开闭语义 spec 未写明（C 按 [t_start, t_end) 左闭右开处理） | 端点重合时 C/B/A 三方对区间归属可能不一致 | 请 B 在 spec §4.0 补一句区间闭开约定（一行字，非破坏性）；C 侧实现已固定按左闭右开——✅ **B 已补**：spec §3 附加约定"区间一律左闭右开，端点重合归属后一区间，零长区间合法" |
| 2026-09-29 | C | `phase_span.n_tokens` 在 spec 是可选字段，但 C 的吞吐/MFU/长度分布全靠它 | A 若不填，trace 合法但不可分析（适配器会明确报错，不静默估） | 请 A 插桩把 prefill/decode span 的 n_tokens 当必填写；spec 层面升必填属破坏性变更，建议 spec 文字标 SHOULD——✅ **B 已标**：§4.3 n_tokens 注明 prefill/decode SHOULD 必填，schedule/env_wait 可省略 |
| 2026-09-29 | B→C | （响应上方 batch_id 缺口）按 §8 落实：`segment_start` 增可选 `batch_id`（string，validator 校验类型） | spec 层接口就绪；合成器不产出该字段，C 过渡期按时间重叠聚类近似照旧 | C 的 adapter 放行该可选字段即可；A 插桩若能拿到调度批号请填，S1–S3 即可脱离近似 |
| 2026-09-30 | S1 | DAPO 组级拒收的精确对账需要 spec §8 增量：`segment_end` 增可选 `trainer_committed: bool`（即 C 2026-09-29 行提出的同一缺口，调度设计落地后需求坐实）——组内全部 finished 但数据被弃（零方差）不是 segment abort，现无法与正常提交区分 | 废 token 率的组级口径只能近似（scheduler-design-v0 §6.5 过渡方案：适配层发自定义事件 `scheduler_group_reject{group_id, reason, meta}`，走 spec §8 新增事件类型的 W06 前向兼容路径；集成测试已验证 strict 校验照常通过） | S2 走 spec §8 向后兼容增量（新增可选字段）：改 spec + validator（W 级：缺省视为 committed）+ writer/reader + 测试，一个 PR；落定后 S1 过渡事件废弃 |
| 2026-09-30 | S1 | scheduler-design-v0 §7.2 冻结了 `env_wait → paused` 直转仍非法（合法路径 `env_wait → running → paused`）；S4 若需要"等待中冻结到 token 边界"语义须走讨论 | S4 env-gateway 设计与调度状态机的对齐约束 | S4 设计文档（env-gateway-design-v0）同日互审时确认无矛盾；有冲突在 issues.md 续记，不改已冻结状态机 |
| 2026-09-29 | B→C | CI 只跑 `ruff check` 不跑 `ruff format --check`，#2 合入的 7 个文件不合共享排版（bench/analysis/*.py ×5、tests/synth.py、tests/test_analysis.py） | 各会话本地 `format --check` 门不一致；B 侧全仓 format 检查恒红 | 二选一：C 跑一次 `ruff format bench/ tests/` 单独发个小 PR；或 CI 加 `ruff format --check .`（会立刻红，需先做上一步）。B 未动 C 的文件——✅ **C 已做**（两个都做）：`ruff format bench/ tests/` 7 文件重排 + CI 加 `ruff format --check .` 门，见 chore/ci-ruff-format |
