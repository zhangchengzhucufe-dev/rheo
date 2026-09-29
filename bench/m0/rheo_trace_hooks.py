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
- AgentLoopManager.generate_sequences    -> engine-level schedule span per
  step (plain function: the original is @auto_await and fit() calls it sync)
- SingleTurnAgentLoop.run                -> segment lifecycle + decode span +
  token_logprob chunk; the LLMServerClient.generate timing is captured via a
  ContextVar so concurrent per-sample tasks don't stomp each other.

No-op (zero imports beyond os) unless RHEO_TRACE=1.
"""

import json
import os
import time
from pathlib import Path

TRACE_DIR = Path(os.environ.get("RHEO_TRACE_DIR", "/tmp/rheo-trace"))
FORMAT = "rheo-trace-spill-1"


def _spill_path() -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return TRACE_DIR / f"spill-{os.getpid()}.jsonl"


def spill(type_: str, **fields: dict) -> None:
    """Append one event to this process's spill file. Never raises into verl."""
    try:
        ev = {"type": type_, "ts": time.time_ns(), "fmt": FORMAT, "pid": os.getpid()}
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


def _install_gen_hook():
    import inspect

    import verl.experimental.agent_loop.agent_loop as al

    # ---- engine-level schedule span around each generation step -----------
    # 原方法带 @auto_await：fit() 是同步调用（无事件循环时内部 asyncio.run），
    # 钩子必须是普通函数；异步上下文时包装 coroutine 保住完成时序
    _orig_mgr_gen = al.AgentLoopManager.generate_sequences

    def traced_mgr_gen(self, prompts):
        now = _now()
        prev = getattr(self, "_rheo_prev_gen_end", None)
        if prev is not None and now >= prev:
            spill("phase_span", phase="schedule", t_start=prev, t_end=now)
        self._rheo_prev_gen_end = None
        result = _orig_mgr_gen(self, prompts)
        if inspect.iscoroutine(result):

            async def _afinish():
                out = await result
                self._rheo_prev_gen_end = _now()
                return out

            return _afinish()
        self._rheo_prev_gen_end = _now()
        return result

    al.AgentLoopManager.generate_sequences = traced_mgr_gen
    print("[rheo-trace] hook: schedule spans (AgentLoopManager)", flush=True)


_GEN_CV = None  # lazily created ContextVar (created inside install())


def _install_gen_timing_hook():
    """LLMServerClient.generate 计时（类级，只包一次）。

    每个 sample 的 run() 在自己的 asyncio.Task 里执行，通过 ContextVar 把
    计时槽传给共享的 wrapper——并发 sample 互不踩踏。
    """
    import contextvars

    import verl.workers.rollout.llm_server as lls

    global _GEN_CV
    if _GEN_CV is None:
        _GEN_CV = contextvars.ContextVar("rheo_gen_timing", default=None)

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


def _install_segment_hook():
    import verl.experimental.agent_loop.single_turn_agent_loop as stl

    _orig_loop_run = stl.SingleTurnAgentLoop.run

    async def traced_loop_run(self, sampling_params: dict, **kwargs):
        from uuid import uuid4

        t_seg_start = _now()
        seg_id = f"s-{uuid4().hex[:12]}"
        info = kwargs.get("extra_info") or {}
        group_id = f"g-{info.get('index', id(kwargs.get('raw_prompt')) & 0xFFFF):08d}"

        gen = {"t0": None, "t1": None}
        _GEN_CV.set(gen)
        output = await _orig_loop_run(self, sampling_params, **kwargs)

        try:
            n_prompt = len(output.prompt_ids)
            n_gen = len(output.response_ids)
            spill(
                "segment_start",
                seg_id=seg_id,
                group_id=group_id,
                ts=t_seg_start,
                t_start=t_seg_start,
                n_prompt_tokens=n_prompt,
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
                spill(
                    "token_logprob",
                    seg_id=seg_id,
                    start_idx=0,
                    n=len(lps),
                    lp=[float(x) for x in lps],
                )
            spill(
                "segment_end",
                seg_id=seg_id,
                state="finished",
                from_state="running",
                t_end=max(_now(), gen["t1"] or t_seg_start),
                n_gen_tokens=n_gen,
            )
        except Exception:
            pass
        return output

    stl.SingleTurnAgentLoop.run = traced_loop_run
    print("[rheo-trace] hook: segment lifecycle (SingleTurnAgentLoop)", flush=True)


def install() -> None:
    for name, fn in [
        ("weight_sync", _install_weight_sync_hook),
        ("schedule", _install_gen_hook),
        ("segment", _install_segment_hook),
        ("generate timing", _install_gen_timing_hook),
    ]:
        try:
            fn()
        except Exception:
            import traceback

            print(f"[rheo-trace] hook {name} FAILED:", flush=True)
            traceback.print_exc()
    spill("trace_boot", pid=os.getpid())


install()
