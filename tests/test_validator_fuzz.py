"""第十一轮：validator 模糊测试——不变式：任何变异输入只允许域内异常逃逸。

完整 fuzz（2 万次）曾抓到 finish_mode 塞 list 时裸抛 TypeError: unhashable；
本文件用 seeded 小规模版本守住这一类回归（全部异常必须是 ValidationError）。
"""

import copy
import random

import rheotrace
from rheotrace.core import RheotraceError, ValidationError

MUT_VALUES = [None, True, "x", -1, 10**25, [], {}, 0.5, ["a"], {"k": 1}]


def _mutate(rng: random.Random, ev: dict) -> dict:
    e = dict(ev)
    op = rng.random()
    if op < 0.3 and e["type"] != "run_start":
        keys = [k for k in e if k != "type"]
        if keys:
            del e[rng.choice(keys)]
    elif op < 0.6:
        keys = [k for k in e if isinstance(e[k], (int, str, list))]
        if keys:
            e[rng.choice(keys)] = rng.choice(MUT_VALUES)
    elif op < 0.8:
        for k in ("ts", "t_start", "t_end"):
            if k in e:
                e[k] = rng.choice([0, -1, 10**18, 10**30])
    else:
        for k in ("state", "from_state", "to_state", "phase", "mode", "finish_mode", "clock"):
            if k in e:
                e[k] = rng.choice(["running", "zzz", "", None, 5, [], {}])
    return e


def test_validator_fuzz_no_escaped_exceptions():
    """2 万次 fuzz 的 seeded 缩减版：除域内异常外零逃逸。"""
    rng = random.Random(20260929)
    base = rheotrace.generate(preset="agent", seed=0)
    for i in range(1500):
        ev = copy.deepcopy(base)
        for _ in range(rng.randint(1, 4)):
            ev[rng.randrange(len(ev))] = _mutate(rng, ev[rng.randrange(len(ev)) if ev else 0])
        try:
            rheotrace.validate(ev, strict=(i % 2 == 0))
        except (ValidationError, TypeError):
            pass  # ValidationError=域内拒绝；TypeError=单事件 dict 误用守卫
        except RheotraceError as e:
            if "单个事件 dict" not in str(e):
                raise AssertionError(f"非误用类 RheotraceError 逃逸: {e}") from None


def test_finish_mode_unhashable_is_e06():
    """fuzz 抓到的具体案例：finish_mode 塞 list/dict 应报 E06 而非裸抛 TypeError。"""
    ev = rheotrace.generate(preset="grpo", seed=0)
    i = next(i for i, e in enumerate(ev) if e["type"] == "segment_end")
    for bad in ([], {}, ["exact"]):
        mutated = [dict(e) for e in ev]
        mutated[i]["finish_mode"] = bad
        rep = rheotrace.validate(mutated, strict=False)
        assert any(x.rule == "E06" and "finish_mode" in x.message for x in rep.errors)
