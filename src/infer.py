"""Generate answers for a split with the base model or base + LoRA adapter.

Baseline:   python src/infer.py --data_dir DATA --split test --system baseline --out OUT/predictions_test.csv
Fine-tuned: python src/infer.py --data_dir DATA --split test --system finetuned --adapter RUN/adapter --out RUN/eval/predictions_test.csv
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GEN_KWARGS, MODEL_ID, PROMPT_VERSION, SEED, build_messages, env_info,
                    load_model_and_tokenizer, load_split, stop_token_ids, write_json)


def clean(text: str) -> str:
    """Keep the first non-empty line; the task asks for one short sentence."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[0] if lines else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", required=True, help="label stored in the output, e.g. baseline / finetuned")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    df = load_split(a.data_dir, a.split)
    if a.limit:
        df = df.head(a.limit)
    model, tok, dtype = load_model_and_tokenizer(a.model_id, a.adapter, padding_side="left")
    model.eval()
    stops = stop_token_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    raw, t0 = [], time.time()
    for i in range(0, len(df), a.batch_size):
        chunk = df.iloc[i:i + a.batch_size]
        texts = [tok.apply_chat_template(build_messages(r.part2, r.question), tokenize=False,
                                         add_generation_prompt=True) for r in chunk.itertuples()]
        # the chat template already contains <bos>; do not add it twice
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        enc = {k: enc[k] for k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = model.generate(**enc, **GEN_KWARGS, eos_token_id=stops, pad_token_id=pad_id)
        new = out[:, enc["input_ids"].shape[1]:]
        raw += tok.batch_decode(new, skip_special_tokens=True)
        print(f"{min(i + a.batch_size, len(df))}/{len(df)}", flush=True)

    res = df[["text_id", "direction", "question", "answer"]].copy()
    res["prediction"] = [clean(t) for t in raw]
    res["raw_output"] = raw
    res["system"] = a.system
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res.to_csv(a.out, index=False)
    write_json({
        "system": a.system, "split": a.split, "n": len(res), "model_id": a.model_id, "adapter": a.adapter,
        "prompt_version": PROMPT_VERSION, "gen_kwargs": GEN_KWARGS, "dtype": str(dtype),
        "seconds": round(time.time() - t0, 1), "empty_predictions": int((res.prediction == "").sum()),
        "env": env_info(),
    }, os.path.splitext(a.out)[0] + "_meta.json")
    print(f"saved {len(res)} predictions to {a.out}")
    print(res[["question", "answer", "prediction"]].head(5).to_string())


if __name__ == "__main__":
    main()
