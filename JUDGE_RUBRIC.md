# Judge rubric (v1)

The judge scores every candidate answer on the test set for both systems, using the same rubric. These are the same rules filter 2 used to build the dataset (match / partial / miss).

## What the judge sees

One row per item in `judge_items.csv`:

| column | meaning |
|---|---|
| `item_id` | anonymous id (J0001...) |
| `direction` | `past` (answer is in part1) or `future` (answer is in part3) |
| `part1`, `part2`, `part3` | the full story. The model only saw part2. |
| `question` | the question |
| `gold_answer` | reference answer |
| `candidate_answer` | the model's answer (first line of its output) |

The judge does **not** see which system wrote the answer. Rows are shuffled, and about 10% are hidden duplicates, which are used to measure the judge's self-consistency.

## Scores

| score | label | rule |
|---|---|---|
| **2** | correct | Same core fact as the gold answer **or** as the hidden sentence (part1 for past, part3 for future). Paraphrases, synonyms and extra detail are fine as long as nothing contradicts the story. |
| **1** | partial | Compatible with the hidden part but clearly less specific, or gets one of two required pieces, or has the right direction with a weaker or different nuance ("mutluydu" vs "çok gurur duydu"). |
| **0** | wrong | Contradicts the hidden part, names a different fact, is a plausible guess the story does not support, is generic or non-informative ("bir şey oldu"), only repeats part2, is empty, or is not Turkish. |

### Rules
1. **Judge against the story, not against plausibility.** A sensible answer that the hidden sentence does not support scores 0.
2. **Length earns nothing.** Extra wording neither helps nor hurts unless it adds a wrong fact. A wrong added fact makes the answer 1 at most.
3. **Ignore grammar and spelling** unless they change the meaning.
4. **Hedged answers** ("X ya da Y") get the score of the worst option.
5. **Yes/no questions:** the polarity must match. A correct polarity with a wrong reason scores 1.

## Output

`judge_scores.csv`, one row per `item_id`:

```
item_id,score,reason
J0001,2,aynı olgu
J0002,0,farklı olay: sahile değil göle gitti
```

`reason` is optional and at most about 10 words. `src/score.py` rejects the file if any item is missing or any score is not 0, 1 or 2.

## Final numbers (computed by `src/score.py`)

- **Judged accuracy** (primary) = (correct + 0.5 × partial) / N
- **Strict accuracy** = correct / N
- **Lenient accuracy** = (correct + partial) / N
- Each metric is reported overall, for past, and for future.
- **Comparison:** the difference in judged accuracy in points, with a 95% paired bootstrap CI. Relative gain is also reported. McNemar's exact test is run on strict correctness.
- **Secondary string metrics:** ROUGE-L F1 and chrF against the gold answer.
- **Judge self-consistency:** agreement on the hidden duplicates.

The "+10%" target is reported both ways, as +10 points absolute and as +10% relative.
