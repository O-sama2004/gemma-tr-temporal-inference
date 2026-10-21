"""Turn judge scores into the final numbers.

python src/score.py --judge_dir EVAL/judge_round1 --scores EVAL/judge_round1/judge_scores.csv \
    --baseline baseline --compare finetuned

judge_scores.csv columns: item_id, score (2 = correct, 1 = partial, 0 = wrong), reason (optional)
Outputs report.md, report.json and per_item.csv in --judge_dir.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- string metrics (secondary)
def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def tokens(s: str) -> list:
    return re.findall(r"\w+", tr_lower(str(s)))


def rouge_l_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def chrf(preds, refs) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF().corpus_score(list(preds), [list(refs)]).score


# ---------------------------------------------------------------- stats
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 42):
    """95% CI of mean(b) - mean(a) over paired items."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray):
    from scipy.stats import binomtest
    b01 = int(((~a_ok) & b_ok).sum())   # baseline wrong, compare right
    b10 = int((a_ok & ~b_ok).sum())     # baseline right, compare wrong
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return b01, b10, float(p)


def summarize(g: pd.DataFrame) -> dict:
    return {
        "n": int(len(g)),
        "judged_accuracy": float((g.score / 2).mean()),          # correct=1, partial=0.5
        "strict_accuracy": float((g.score == 2).mean()),         # correct only
        "lenient_accuracy": float((g.score >= 1).mean()),        # correct or partial
        "correct": int((g.score == 2).sum()), "partial": int((g.score == 1).sum()), "wrong": int((g.score == 0).sum()),
        "rougeL_f1": float(np.mean([rouge_l_f1(p, r) for p, r in zip(g.candidate_answer, g.gold_answer)])),
        "chrF": chrf(g.candidate_answer.astype(str), g.gold_answer.astype(str)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_dir", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--compare", default="finetuned")
    a = ap.parse_args()

    items = pd.read_csv(f"{a.judge_dir}/judge_items.csv", keep_default_na=False)
    key = pd.read_csv(f"{a.judge_dir}/judge_key.csv", keep_default_na=False)
    sc = pd.read_csv(a.scores, keep_default_na=False)
    assert set(sc.item_id) == set(items.item_id), "scores must cover every judge item exactly once"
    assert sc.item_id.is_unique
    sc["score"] = sc.score.astype(int)
    assert sc.score.isin([0, 1, 2]).all(), "score must be 0, 1 or 2"
    df = items.merge(key, on=["item_id", "direction"]).merge(sc[["item_id", "score"]], on="item_id")

    # judge self-consistency on hidden duplicates
    dups = df[df.dup_of != ""]
    orig = df.set_index("item_id").score
    cons = {"n_duplicates": int(len(dups))}
    if len(dups):
        same = (dups.score.values == orig.loc[dups.dup_of].values)
        cons.update(exact_agreement=float(same.mean()),
                    keep_drop_agreement=float(((dups.score.values >= 1) == (orig.loc[dups.dup_of].values >= 1)).mean()))
    df = df[df.dup_of == ""].copy()

    report = {"judge_consistency": cons, "systems": {}}
    for name, g in df.groupby("system"):
        report["systems"][name] = {"overall": summarize(g),
                                   **{d: summarize(gg) for d, gg in g.groupby("direction")}}

    if a.baseline in report["systems"] and a.compare in report["systems"]:
        b = df[df.system == a.baseline].set_index(["text_id", "direction"]).sort_index()
        c = df[df.system == a.compare].set_index(["text_id", "direction"]).sort_index()
        assert b.index.equals(c.index), "systems were not judged on the same items"
        bs, cs = (b.score / 2).values, (c.score / 2).values
        lo, hi = paired_bootstrap(bs, cs)
        b01, b10, p = mcnemar_exact(b.score.values == 2, c.score.values == 2)
        base_acc, comp_acc = bs.mean(), cs.mean()
        report["comparison"] = {
            "baseline": a.baseline, "compare": a.compare,
            "judged_accuracy_baseline": float(base_acc), "judged_accuracy_compare": float(comp_acc),
            "delta_points": float(100 * (comp_acc - base_acc)),
            "delta_points_95ci": [100 * lo, 100 * hi],
            "relative_gain_percent": float(100 * (comp_acc - base_acc) / base_acc) if base_acc else None,
            "mcnemar_strict": {"baseline_wrong_compare_right": b01, "baseline_right_compare_wrong": b10, "p": p},
            "meets_10_points": bool(100 * (comp_acc - base_acc) >= 10),
        }

    per_item = df[["item_id", "system", "text_id", "direction", "question", "gold_answer", "candidate_answer", "score"]]
    per_item.to_csv(f"{a.judge_dir}/per_item.csv", index=False)
    json.dump(report, open(f"{a.judge_dir}/report.json", "w"), indent=2, ensure_ascii=False)

    lines = ["# Judged evaluation", ""]
    lines += ["| system | split | n | judged acc | strict | lenient | ROUGE-L | chrF |", "|---|---|---|---|---|---|---|---|"]
    for name, s in report["systems"].items():
        for part, m in s.items():
            lines.append(f"| {name} | {part} | {m['n']} | {100*m['judged_accuracy']:.1f} | {100*m['strict_accuracy']:.1f} | "
                         f"{100*m['lenient_accuracy']:.1f} | {100*m['rougeL_f1']:.1f} | {m['chrF']:.1f} |")
    if "comparison" in report:
        c = report["comparison"]
        lines += ["", f"**{c['compare']} vs {c['baseline']}:** {c['delta_points']:+.1f} points judged accuracy "
                  f"(95% CI {c['delta_points_95ci'][0]:+.1f} to {c['delta_points_95ci'][1]:+.1f}), "
                  f"relative {c['relative_gain_percent']:+.1f}%. McNemar (strict) p = {c['mcnemar_strict']['p']:.4f}. "
                  f"Meets +10 points: {c['meets_10_points']}."]
    if cons["n_duplicates"]:
        lines += ["", f"Judge self-consistency on {cons['n_duplicates']} hidden duplicates: "
                  f"{100*cons['exact_agreement']:.1f}% same score, {100*cons['keep_drop_agreement']:.1f}% same pass/fail."]
    open(f"{a.judge_dir}/report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
