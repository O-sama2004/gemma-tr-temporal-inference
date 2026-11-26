# gemma-tr-temporal-inference

Dil Modellerinde Bağlamdan Geçmiş-Gelecek Çıkarımı. We fine-tune `google/gemma-3-1b-it` with LoRA so that, given only the middle of a Turkish story (part 2), it answers a question about what happened before (past) or after (future), and answers "Bilmiyorum." when part 2 does not give enough clues. We compare the untrained and the fine-tuned model on the same test items using judged accuracy and abstention F1.

## Results so far

| data | test | baseline | LoRA fine-tuned | difference |
|---|---|---|---|---|
| v1 (4,686 train pairs) | 350 pairs | 14.4 judged acc | 25.3 | +10.9 points (95% CI +7.3 to +14.4) |
| v3 (8,063 train pairs) | same 350 pairs | | | +17.6 points (95% CI +13.4 to +21.7) |
| v4 (9,849 train items, 20% "Bilmiyorum.") | 668 items (534 answerable, 134 "Bilmiyorum.") | pending | pending | |

v4 changes the prompt (`p2`) and the test set, so its numbers are not directly comparable with v1 and v3. The 350 v3 test pairs are inside the v4 test set, and notebook 03 reports them separately.

## Layout

```
notebooks/
  01_baseline.ipynb          untrained model -> eval/baseline_untrained_v4/predictions_test.csv
  02_finetune.ipynb          train -> runs/<RUN_ID>/ (adapter, run.json, loss.png) -> test predictions
  03_judge_and_score.ipynb   blind judge file -> (judge) -> report.md, results.png
JUDGE_RUBRIC.md              scoring rules (2 correct, 1 partial, 0 wrong) and the automatic abstention scores
```

Each notebook is self-contained: all code, pinned library versions and dataset checksums are inside it. Open a notebook in Colab with File > Open notebook > GitHub.

Data and outputs live in Google Drive, never in this repo (ROCStories research-use terms):

```
MyDrive/gemma-tr-temporal-inference/
  data/v4/        splits/train.csv, splits/test.csv, structured/structured.csv, README.md (older versions in data/v1, data/v3)
  eval/baseline_untrained_v4/   predictions_test.csv, run.json
  eval/finetuned_v3_on_v4/      optional: the v3 adapter on the v4 test set
  eval/judge_<RUN_ID>/          judge_items.csv, judge_key.csv + all_predictions.csv (private), judge_scores.csv,
                                report.md, report.json, per_item.csv, results.png
  runs/<RUN_ID>/                adapter/ (adapter_model.safetensors, adapter_config.json), run.json, loss.png, predictions_test.csv
  ckpt/<RUN_ID>/                checkpoints while training (deleted when the adapter is saved)
```

## One-time setup

1. Upload the dataset files above to `MyDrive/gemma-tr-temporal-inference/data/v4/`.
2. Accept the Gemma licence at https://huggingface.co/google/gemma-3-1b-it and create a **read** token.
3. In Colab, open the key icon, add a secret named `HF_TOKEN`, and enable notebook access.

## Run order

| step | notebook | GPU | time (L4) | output |
|---|---|---|---|---|
| 1 | `01_baseline` | yes | ~8 min | baseline predictions on test |
| 2 | `02_finetune` | yes | ~65 min train + ~5 min inference | adapter + fine-tuned predictions on test |
| 2b | `02_finetune`, optional cell | yes | ~5 min | v3 adapter on the v4 test set |
| 3 | `03_judge_and_score` step 1 | no | <1 min | `judge_items.csv` (send to the judge) |
| 4 | judge | n/a | n/a | `judge_scores.csv` (put next to `judge_items.csv`) |
| 5 | `03_judge_and_score` step 2 | no | ~1 min | `report.md`, `report.json`, `results.png` |

On a T4 the code switches to fp32, because Gemma 3 is unstable in fp16. Training then takes about 2.5 times longer. Training resumes from Drive checkpoints after a disconnect (see the notebook header).

## What is fixed for a fair comparison

- The same prompt (version `p2`) and the same greedy decoding with 64 new tokens for every system. The "Shared settings" cell is identical in notebooks 01 and 02.
- The same 668 test items. No test story is used for training or for choosing checkpoints. The dev set is 10% of the train stories (seed 42).
- The torch version, GPU and dtype are logged in every `run.json`.

## Metrics

- **Overall score** (primary for v4): mean over all test items of score / 2. Answerable item: judge score 2 / 1 / 0, and 0 if the model abstained. "Bilmiyorum." item: 2 if the model abstained, else 0. An abstention is caught by a pattern first; the judge flags any other wording (`abstains` column), so a paraphrased "I don't know" is not scored as wrong.
- **Judged accuracy on answerable items** = (correct + 0.5 x partial) / N, also for past, future and the 350 v3 test pairs. Scored blind by a stronger model using `JUDGE_RUBRIC.md`. About 10% of judged items are hidden duplicates.
- **Abstention** precision, recall and F1, recall per source (`filter2_drop`, `new_not_in_story`), false abstention rate on answerable items, and recall by question type. A cue-only rule (abstain on hangi / kaç / ne kadar questions) gives F1 0.37 on test; report model F1 next to it.
- Differences with 95% paired bootstrap CIs, McNemar's p on strict correctness, ROUGE-L and chrF as secondary metrics.

The code that produced the v1 numbers (with `src/`) is tagged `v1-results`.
