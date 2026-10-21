"""Build one blind file for the judge from several systems' predictions.

python src/make_judge_file.py --data_dir DATA \
    --pred baseline=EVAL/baseline_untrained/predictions_test.csv \
    --pred finetuned=RUN/eval/predictions_test.csv \
    --out_dir EVAL/judge_round1

Writes
  judge_items.csv  -> send to the judge. No system names, rows shuffled.
  judge_key.csv    -> keep private. Maps item_id to system; needed by score.py.
A share of items (--dup_frac) is repeated under new ids to measure judge self-consistency.
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=path, repeat per system")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dup_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ref = pd.read_csv(f"{a.data_dir}/splits/{a.split}.csv")
    st = pd.read_csv(f"{a.data_dir}/structured/structured.csv")[["text_id", "part1", "part3"]]
    ref = ref.merge(st, on="text_id", validate="many_to_one")
    rows = []
    for spec in a.pred:
        name, path = spec.split("=", 1)
        p = pd.read_csv(path, keep_default_na=False)[["text_id", "direction", "prediction"]]
        m = ref.merge(p, on=["text_id", "direction"], how="left", validate="one_to_one")
        miss = m.prediction.isna().sum()
        assert miss == 0, f"{name}: {miss} test items have no prediction"
        m["system"] = name
        rows.append(m)
    allp = pd.concat(rows, ignore_index=True)

    rng = np.random.default_rng(a.seed)
    allp["uid"] = range(len(allp))
    allp["dup_uid"] = -1
    n_dup = int(round(a.dup_frac * len(allp)))
    if n_dup:
        dup = allp.iloc[rng.choice(len(allp), n_dup, replace=False)].copy()
        dup["dup_uid"] = dup["uid"]
        dup["uid"] = range(len(allp), len(allp) + n_dup)
        allp = pd.concat([allp, dup], ignore_index=True)
    allp = allp.iloc[rng.permutation(len(allp))].reset_index(drop=True)
    allp["item_id"] = [f"J{i + 1:04d}" for i in range(len(allp))]
    uid2item = dict(zip(allp.uid, allp.item_id))
    allp["dup_of"] = allp.dup_uid.map(lambda u: uid2item[u] if u >= 0 else "")

    os.makedirs(a.out_dir, exist_ok=True)
    allp[["item_id", "direction", "part1", "part2", "part3", "question", "answer", "prediction"]] \
        .rename(columns={"answer": "gold_answer", "prediction": "candidate_answer"}) \
        .to_csv(f"{a.out_dir}/judge_items.csv", index=False)
    allp[["item_id", "system", "text_id", "direction", "dup_of"]].to_csv(f"{a.out_dir}/judge_key.csv", index=False)
    print(f"{len(allp)} judge items ({len(allp) - n_dup} unique + {n_dup} hidden duplicates) "
          f"from {len(a.pred)} systems -> {a.out_dir}")
    print("send judge_items.csv only; keep judge_key.csv private")


if __name__ == "__main__":
    main()
