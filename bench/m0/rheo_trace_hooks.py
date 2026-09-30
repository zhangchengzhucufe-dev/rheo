"""RheoTrace instrumentation for verl 0.8.0 (TASK-A step 3).

Loaded via a ``sitecustomize.py`` bootstrap in ~/tools/venvs/rheo whenever
``RHEO_TRACE=1``, ``RHEO_TRACE_HOOKS`` points at this file, and argv[0] is not
a ``-c`` one-shot. Every ray worker process gets the monkey patches below;
events are appended to per-process spill files
(``$RHEO_TRACE_DIR/spill-<pid>.jsonl``) with wall-clock ns timestamps, then
merged into a single RheoTrace file by ``bench/m0/merge_trace.py`` (spec §3
sanctions per-worker files; the merger replays the weight_sync ledger to
derive per-segment birth/end_version).

Patched sites:
- CheckpointEngineManager.update_weights -> weight_sync (driver side; worker
  side carries @register dispatch metadata that a plain function would break)
(schedule spans are derived at merge time from segment boundaries — the
manager-level hook for them kept silently not firing and was removed)
- SingleTurnAgentLoop.run                -> segment lifecycle + decode span +
  token_logprob chunk; the LLMServerClient.generate timing is captured via a
  ContextVar so concurrent per-sample tasks don't stomp each other
- AgentLoopWorker.generate_sequences     -> stash meta_info validate/global_steps
  into a ContextVar; per-sample tasks inherit it (create_task copies context),
  letting segment meta distinguish validation from training samples

No-op (stdlib-only imports) unless RHEO_TRACE=1.
"""

import contextvars
import json
import math
import os
import time
from pathlib import Path

TRACE_DIR = Path(os.environ.get("RHEO_TRACE_DIR", "/tmp/rheo-trace"))
# 每次启动一个 run_id，spill 按 run_id 分子目录（TASK-A2 D14/G1）：
# 崩溃续跑时事件续写进同一 run 目录；merge 按 run_id 合并，绝不混入历史 run
RUN_ID = os.environ.get("RHEO_RUN_ID", "r-unknown")
FORMAT = "rheo-trace-spill-1"

_GEN_CV: contextvars.ContextVar = contextvars.ContextVar("rheo_gen_timing", default=None)
_WORKER_CV: contextvars.ContextVar = contextvars.ContextVar("rheo_worker_ctx", default=None)


def _spill_path() -> Path:
    (TRACE_DIR / RUN_ID).mkdir(parents=True, exist_ok=True)
    return TRACE_DIR / RUN_ID / f"spill-{os.getpid()}.jsonl"


def spill(type_: str, **fields: dict) -> None:
    """Append one event to this process's spill file. Never raises into verl."""
    try:
        ev = {
            "type": type_,
            "ts": time.time_ns(),
            "fmt": FORMAT,
            "pid": os.getpid(),
            "run_id": RUN_ID,
        }
        ev.update(fields)
        with open(_spill_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:  # telemetry must never kill training
        pass


def _now() -> int:
    return time.time_ns()


def _install_weight_sync_hook():
    import verl.checkpoint_engine.base as ceb

    _orig_update_weights = ceb.CheckpointEngineManager.update_weights

    def traced_update_weights(self, global_steps: int = None):
        t0 = _now()
        result = _orig_update_weights(self, global_steps=global_steps)
        spill(
            "weight_sync",
            version=int(global_steps or 0),
            t_start=t0,
            t_end=_now(),
            mode="full",
            trainer_step=int(global_steps or 0),
        )
        return result

    ceb.CheckpointEngineManager.update_weights = traced_update_weights
    print("[rheo-trace] hook: weight_sync (CheckpointEngineManager)", flush=True)


def _install_gen_timing_hook():
    """LLMServerClient.generate 计时（类级，只包一次）。

    每个 sample 的 run() 在自己的 asyncio.Task 里执行，通过 ContextVar 把
    计时槽传给共享的 wrapper——并发 sample 互不踩踏。
    """

    import verl.workers.rollout.llm_server as lls

    orig_generate = lls.LLMServerClient.generate

    async def timed_generate(self, *a, **kw):
        slot = _GEN_CV.get()
        if slot is not None and slot.get("t0") is None:
            slot["t0"] = _now()
        out = await orig_generate(self, *a, **kw)
        if slot is not None:
            slot["t1"] = _now()
        return out

    timed_generate.__rheo_wrapped__ = True
    lls.LLMServerClient.generate = timed_generate
    print("[rheo-trace] hook: generate timing (LLMServerClient)", flush=True)


def _install_worker_ctx_hook():

    import verl.experimental.agent_loop.agent_loop as al

    _orig_worker_gen = al.AgentLoopWorker.generate_sequences

    async def traced_worker_gen(self, batch):
        # ContextVar 经 sys.modules 取用而非闭包捕获：AgentLoopWorker 是 ray
        # actor 类，ray.remote() 会序列化类定义，闭包里的 ContextVar 不可 pickle
        import sys as _sys

        cv = _sys.modules["rheo_trace_hooks"]._WORKER_CV
        meta = getattr(batch, "meta_info", None) or {}
        tok = cv.set(
            {
                "validate": bool(meta.get("validate", False)),
                "global_steps": meta.get("global_steps", None),
            }
        )
        try:
            return await _orig_worker_gen(self, batch)
        finally:
            cv.reset(tok)

    al.AgentLoopWorker.generate_sequences = traced_worker_gen
    print("[rheo-trace] hook: worker ctx (AgentLoopWorker)", flush=True)


def _install_segment_hook():
    import verl.experimental.agent_loop.single_turn_agent_loop as stl

    _orig_loop_run = stl.SingleTurnAgentLoop.run

    async def traced_loop_run(self, sampling_params: dict, **kwargs):
        from uuid import uuid4

        t_seg_start = _now()
        seg_id = f"s-{uuid4().hex[:12]}"
        # group_id 解析失败绝不能把异常传进 verl（会杀掉样本），降级到未知组
        try:
            info = kwargs.get("extra_info") or {}
            group_id = f"g-{int(info.get('index')):08d}"
        except Exception:
            group_id = "g-unknown"

        gen = {"t0": None, "t1": None}
        _GEN_CV.set(gen)
        output = await _orig_loop_run(self, sampling_params, **kwargs)

        try:
            wctx = _WORKER_CV.get() or {}
            seg_meta = {
                k: v
                for k, v in (
                    ("validate", wctx.get("validate")),
                    ("global_steps", wctx.get("global_steps")),
                )
                if v is not None
            }
            n_prompt = len(output.prompt_ids)
            n_gen = len(output.response_ids)
            spill(
                "segment_start",
                seg_id=seg_id,
                group_id=group_id,
                ts=t_seg_start,
                t_start=t_seg_start,
                n_prompt_tokens=n_prompt,
                meta=seg_meta,
            )
            if gen["t0"] is not None and gen["t1"] is not None:
                spill(
                    "phase_span",
                    seg_id=seg_id,
                    phase="decode",
                    t_start=gen["t0"],
                    t_end=gen["t1"],
                    n_tokens=n_gen,
                )
            lps = getattr(output, "response_logprobs", None)
            if lps:
                # -inf/NaN 会写出非法 JSON（rheotrace allow_nan=False），钳到有限值
                lp = [float(x) if math.isfinite(float(x)) else -1e30 for x in lps]
                spill(
                    "token_logprob",
                    seg_id=seg_id,
                    start_idx=0,
                    n=len(lp),
                    lp=lp,
                )
            spill(
                "segment_end",
                seg_id=seg_id,
                state="finished",
                from_state="running",
                t_end=max(_now(), gen["t1"] or t_seg_start),
                n_gen_tokens=n_gen,
                meta=seg_meta,
            )
        except Exception:
            pass
        return output

    stl.SingleTurnAgentLoop.run = traced_loop_run
    print("[rheo-trace] hook: segment lifecycle (SingleTurnAgentLoop)", flush=True)


def install() -> None:
    for name, fn in [
        ("weight_sync", _install_weight_sync_hook),
        ("segment", _install_segment_hook),
        ("generate timing", _install_gen_timing_hook),
        ("worker ctx", _install_worker_ctx_hook),
    ]:
        try:
            fn()
        except Exception:
            import traceback

            print(f"[rheo-trace] hook {name} FAILED:", flush=True)
            traceback.print_exc()
    spill("trace_boot", pid=os.getpid())


install()
