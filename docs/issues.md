# 跨会话问题日志

发现**别人板块**的问题（坑、接口不对、规格缺失）：记在这里，不要顺手实现。
自己板块的 bug 直接修。

| 日期 | 会话 | 发现 | 影响 | 建议 |
|---|---|---|---|---|
| 2026-09-29 | B | PR #2（`feat/analysis`）分支基点过老：diff 含整个仓库骨架，且带 `rheotrace/__init__.py` 的占位版改动 | 与已合入 main 的 rheotrace 包（#3）直接冲突，无法干净 merge | merge 前 rebase 到最新 main；`rheotrace/*` 以 main 为准（占位 docstring 已被真实现取代），删掉本分支对该文件的改动 |
| | | | | |
