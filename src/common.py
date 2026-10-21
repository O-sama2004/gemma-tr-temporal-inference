"""Shared settings, prompt, data loading and model loading.

Everything that must be identical for the baseline and the fine-tuned run lives here:
the model id, the prompt, the decoding settings and the train/dev split.
"""
import json
import os
import platform

import numpy as np
import pandas as pd

MODEL_ID = "google/gemma-3-1b-it"
SEED = 42
DEV_FRAC = 0.10          # share of TRAIN stories held out for eval loss / checkpoint choice
PROMPT_VERSION = "p1"

# Decoding is greedy and fixed so that baseline and fine-tuned outputs are comparable.
GEN_KWARGS = dict(do_sample=False, max_new_tokens=64, repetition_penalty=1.0)

INSTRUCTION = (
    "Aşağıda bir hikâyenin ortasından alınmış bir bölüm var. "
    "Soru, bu bölümden önce ya da sonra olan bir şeyi soruyor. "
    "Bölümdeki ipuçlarını kullanarak en olası cevabı Türkçe ve tek bir kısa cümleyle yaz."
)


def build_messages(part2: str, question: str) -> list:
    """Single user turn (Gemma 3 has no system role)."""
    return [{"role": "user", "content": f"{INSTRUCTION}\n\nMetin: {part2}\n\nSoru: {question}"}]


# ---------------------------------------------------------------- data
def load_split(data_dir: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, "splits", f"{split}.csv"))
    need = {"text_id", "direction", "part2", "question", "answer"}
    missing = need - set(df.columns)
    assert not missing, f"{split}.csv is missing columns {missing}"
    assert df[list(need)].notna().all().all(), f"{split}.csv has empty cells"
    return df


def train_dev_split(train_df: pd.DataFrame, dev_frac: float = DEV_FRAC, seed: int = SEED):
    """Hold out whole stories from train (never from test)."""
    ids = np.array(sorted(train_df.text_id.unique()))
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_dev = int(round(dev_frac * len(ids)))
    dev_ids = set(ids[perm[:n_dev]])
    dev = train_df[train_df.text_id.isin(dev_ids)].reset_index(drop=True)
    tr = train_df[~train_df.text_id.isin(dev_ids)].reset_index(drop=True)
    assert not set(tr.text_id) & set(dev.text_id)
    return tr, dev


def to_prompt_completion(df: pd.DataFrame):
    """Conversational prompt-completion format: TRL computes the loss on the completion only."""
    from datasets import Dataset
    return Dataset.from_list([
        {"prompt": build_messages(r.part2, r.question),
         "completion": [{"role": "assistant", "content": r.answer}]}
        for r in df.itertuples()
    ])


# ---------------------------------------------------------------- model
def pick_dtype():
    """bf16 on Ampere+ (L4, A100). fp32 on T4: Gemma 3 is unstable in fp16."""
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(model_id: str = MODEL_ID, adapter: str = None, padding_side: str = "right"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = pick_dtype()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = padding_side
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="eager")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tok, dtype


def stop_token_ids(tok) -> list:
    ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tok.unk_token_id:
        ids.add(eot)
    return sorted(i for i in ids if i is not None)


def env_info() -> dict:
    import torch, transformers, trl, peft
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
