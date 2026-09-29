"""M0 baseline: build GSM8K parquet files for verl GRPO.

Output goes to $RHEO_DATA_DIR (default ~/datasets/rheo/gsm8k), never the repo.
Prompt format mirrors verl's official preprocessing (messages list + rule-based
reward metadata). An explicit "#### <answer>" instruction is appended because
the gsm8k reward extractor is strict-format; without it an instruct model
starts at all-zero reward and GRPO gets no gradient signal.
"""

import os

import datasets

INSTRUCTION = (
    "\n\nSolve the problem step by step. End your response with the final "
    "numeric answer on its own line in the format: #### <answer>"
)


def make_map_fn(split: str):
    def process_fn(example, idx):
        question = example.pop("question")
        answer = example.pop("answer")
        # GSM8K final answer is the number after the last '####'
        solution = answer.strip().split("####")[-1].strip().replace(",", "")
        return {
            "data_source": "gsm8k",
            "prompt": [{"role": "user", "content": question + INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx, "answer": answer},
        }

    return process_fn


def main() -> None:
    out_dir = os.environ.get("RHEO_DATA_DIR", os.path.expanduser("~/datasets/rheo/gsm8k"))
    os.makedirs(out_dir, exist_ok=True)

    dataset = datasets.load_dataset("openai/gsm8k", "main")
    for split, n_val in (("train", None), ("test", 200)):
        ds = dataset[split].map(
            make_map_fn(split),
            with_indices=True,
            remove_columns=dataset[split].column_names,
            desc=f"gsm8k-{split}",
        )
        if split == "test":
            # small validation slice so the reward curve evaluates fast
            ds = ds.select(range(n_val))
        path = os.path.join(out_dir, f"{split}.parquet")
        ds.to_parquet(path)
        print(f"wrote {len(ds)} rows -> {path}")


if __name__ == "__main__":
    main()
