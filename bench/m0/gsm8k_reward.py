"""GSM8K rule-based reward, wrapped for verl 0.8's reward_manager call signature.

verl 0.8 calls:  compute_score(data_source=..., solution_str=...,
                               ground_truth=..., extra_info=...)
verl's own gsm8k.compute_score has the older (solution_str, ground_truth) form,
so we adapt here instead of relying on the installed signature.
"""

import re


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    method="strict",
    format_score=0.0,
    score=1.0,
):
    assert solution_str is not None and ground_truth is not None

    # verl/utils/reward_score/gsm8k.py logic, verbatim
    solution_str = solution_str[-300:] if len(solution_str) > 300 else solution_str
    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        final_answer = solutions[-1].replace(",", "").replace("$", "") if solutions else None
    else:
        answers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        for candidate in reversed(answers):
            if candidate not in ("", "."):
                final_answer = candidate
                break

    if final_answer is None:
        return 0
    return score if final_answer == ground_truth else format_score
