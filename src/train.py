"""LoRA SFT of gemma-3-1b-it on (part2, question) -> answer.

python src/train.py --data_dir DATA --out_dir RUNS/2026-10-25_lora_r16_lr2e-4
Smoke test: add --max_steps 5 --out_dir /content/smoke
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEV_FRAC, MODEL_ID, PROMPT_VERSION, SEED, env_info, load_model_and_tokenizer,
                    load_split, to_prompt_completion, train_dev_split, write_json)


def show_one_example(trainer, tok):
    """Print what the model sees and which tokens carry loss, so masking can be checked by eye."""
    ex = trainer.train_dataset[0]
    ids = ex["input_ids"]
    mask = ex.get("completion_mask")
    print("---- model input ----")
    print(tok.decode(ids))
    if mask is not None:
        print("---- tokens with loss (should be: <start_of_turn>model + answer + <end_of_turn>) ----")
        print(tok.decode([t for t, m in zip(ids, mask) if m]))
    print("---------------------")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dev_frac", type=float, default=DEV_FRAC)
    ap.add_argument("--max_steps", type=int, default=-1, help=">0 for a smoke test")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    train_all = load_split(a.data_dir, "train")
    tr_df, dev_df = train_dev_split(train_all, a.dev_frac, a.seed)
    pd.DataFrame({"text_id": sorted(dev_df.text_id.unique())}).to_csv(f"{a.out_dir}/dev_ids.csv", index=False)
    print(f"train {len(tr_df)} pairs / {tr_df.text_id.nunique()} stories | "
          f"dev {len(dev_df)} pairs / {dev_df.text_id.nunique()} stories")

    model, tok, dtype = load_model_and_tokenizer(a.model_id, padding_side="right")
    model.config.use_cache = False
    smoke = a.max_steps > 0
    cfg = SFTConfig(
        output_dir=f"{a.out_dir}/checkpoints",
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=1 if smoke else 10,
        eval_strategy="steps" if smoke else "epoch",
        save_strategy="steps" if smoke else "epoch",
        eval_steps=a.max_steps if smoke else None,
        save_steps=a.max_steps if smoke else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        max_length=a.max_length,
        packing=False,
        completion_only_loss=True,
        report_to="none",
        seed=a.seed,
    )
    peft_cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                          target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=to_prompt_completion(tr_df),
                         eval_dataset=to_prompt_completion(dev_df), processing_class=tok, peft_config=peft_cfg)
    trainer.model.print_trainable_parameters()
    show_one_example(trainer, tok)

    trainer.train()
    final_eval = trainer.evaluate()
    trainer.save_model(f"{a.out_dir}/adapter")      # LoRA weights only (best checkpoint by dev loss)
    tok.save_pretrained(f"{a.out_dir}/adapter")
    pd.DataFrame(trainer.state.log_history).to_csv(f"{a.out_dir}/log_history.csv", index=False)
    write_json({
        "args": vars(a), "model_id": a.model_id, "prompt_version": PROMPT_VERSION, "dtype": str(dtype),
        "n_train_pairs": len(tr_df), "n_dev_pairs": len(dev_df),
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_dev_loss": trainer.state.best_metric,
        "final_eval": final_eval, "env": env_info(),
    }, f"{a.out_dir}/run_config.json")
    print(f"adapter saved to {a.out_dir}/adapter | best dev loss {trainer.state.best_metric}")


if __name__ == "__main__":
    main()
