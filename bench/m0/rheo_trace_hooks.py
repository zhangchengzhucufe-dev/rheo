"""RheoTrace instrumentation for verl 0.8.0 (TASK-A step 3).

Loaded via a ``sitecustomize.py`` bootstrap in ~/tools/venvs/rheo whenever
``RHEO_TRACE=1`` and ``RHEO_TRACE_HOOKS`` points at this file. Every ray worker
process that imports verl gets the monkey patches below; events are appended to
per-process spill files (``$RHEO_TRACE_DIR/spill-<pid>.jsonl``) with wall-clock
ns timestamps, then merged into a single RheoTrace file by
``bench/m0/merge_trace.py`` (spec §3 sanctions per-worker files; the merger
replays the weight_sync ledger to derive per-segment birth/end_version).

Patched sites:
- ActorRolloutRefWorker.update_weights  -> weight_sync (version=global_steps)
- AgentLoopManager.generate_sequences   -> engine-level schedule span per step
- SingleTurnAgentLoop.run               -> segment lifecycle + decode span +
                                          token_logprob chunk + submit-wait span

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


def _install_gen_hook():
    import verl.experimental.agent_loop.agent_loop as al

    # ---- engine-level schedule span around each generation step -----------
    _orig_mgr_gen = al.AgentLoopManager.generate_sequences

    async def traced_mgr_gen(self, prompts):
        now = _now()
        prev = getattr(self, "_rheo_prev_gen_end", None)
        if prev is not None and now >= prev:
            spill("phase_span", phase="schedule", t_start=prev, t_end=now)
        self._rheo_prev_gen_end = None
        out = await _orig_mgr_gen(self, prompts)
        self._rheo_prev_gen_end = _now()
        return out

    al.AgentLoopManager.generate_sequences = traced_mgr_gen
    print("[rheo-trace] hook: schedule spans (AgentLoopManager)", flush=True)


def _install_segment_hook():
    import verl.experimental.agent_loop.single_turn_agent_loop as stl

    # ---- per-segment lifecycle: SingleTurnAgentLoop.run --------------------
    _orig_loop_run = stl.SingleTurnAgentLoop.run

    async def traced_loop_run(self, sampling_params: dict, **kwargs):
        from uuid import uuid4

        t_seg_start = _now()
        seg_id = f"s-{uuid4().hex[:12]}"
        info = kwargs.get("extra_info") or {}
        group_id = f"g-{info.get('index', id(kwargs.get('raw_prompt')) & 0xFFFF):08d}"

        # wrap the underlying generate call to bracket decode + capture logprobs
        orig_generate = self.server_manager.generate
        gen = {"t0": None, "t1": None, "out": None}

        async def timed_generate(*a, **kw):
            gen["t0"] = _now()
            out = await orig_generate(*a, **kw)
            gen["t1"] = _now()
            gen["out"] = out
            return out

        self.server_manager.generate = timed_generate
        try:
            output = await _orig_loop_run(self, sampling_params, **kwargs)
        finally:
            self.server_manager.generate = orig_generate

        try:
            n_prompt = len(output.prompt_ids)
            n_gen = len(output.response_ids)
            spill(
                "segment_start",
                seg_id=seg_id,
                group_id=group_id,
                t_start=t_seg_start,
                n_prompt_tokens=n_prompt,
            )
            if gen["t0"] is not None:
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
    ]:
        try:
            fn()
        except Exception:
            import traceback

            print(f"[rheo-trace] hook {name} FAILED:", flush=True)
            traceback.print_exc()
    spill("trace_boot", pid=os.getpid())


def _install_weight_sync_hook():
    import os  # noqa: F401
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


install()
