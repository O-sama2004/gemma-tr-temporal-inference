# Judge rubric (v2, dataset v4)

The judge scores candidate answers on the test set for every system, using the same rubric.

**What changed in v2.** Dataset v4 adds test items whose correct answer is "Bilmiyorum." (part 2 does not give enough clues). Answers that notebook 03's pattern recognises as abstentions ("Bilmiyorum", "belirtilmemiş", "bilgi yok", ...) never reach the judge. Every other answer does, including answers on "Bilmiyorum." items, whose `gold_answer` is `Bilmiyorum.`. The judge gives every item a `score` and an `abstains` flag.

## What the judge sees

One row per item in `judge_items.csv`:

| column | meaning |
|---|---|
| `item_id` | anonymous id (J0001...) |
| `direction` | `past` (answer is in part1), `future` (answer is in part3), or `none` (the story never gives the answer) |
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

### Items whose gold answer is "Bilmiyorum."
The story does not give the answer. Score **2** if the candidate says, in any words, that the story does not give the information ("Hikâyede söylenmiyor", "Bunu bilemeyiz", "Metinden çıkarılamıyor"). Score **0** if it gives any concrete answer, even a plausible one, or a concrete answer with a hedge.

### The `abstains` flag (every item)
**1** if the candidate's main message is that the information is not given or cannot be known, in any wording, whether or not that is correct. **0** if it gives a concrete answer. On an answerable item an abstention scores 0 and gets `abstains = 1`.

### Rules
1. **Judge against the story, not against plausibility.** A sensible answer that the hidden sentence does not support scores 0.
2. **Length earns nothing.** Extra wording neither helps nor hurts unless it adds a wrong fact. A wrong added fact makes the answer 1 at most.
3. **Ignore grammar and spelling** unless they change the meaning.
4. **Hedged answers** ("X ya da Y") get the score of the worst option.
5. **Yes/no questions:** the polarity must match. A correct polarity with a wrong reason scores 1.

## Output

`judge_scores.csv`, one row per `item_id`:

```
item_id,score,abstains,reason
J0001,2,0,aynı olgu
J0002,0,0,farklı olay: sahile değil göle gitti
J0003,2,1,gold Bilmiyorum, aday bilginin verilmediğini söylüyor
```

`reason` is optional and at most about 10 words. Notebook 03 rejects the file if any item is missing, any score is not 0, 1 or 2, or any `abstains` is not 0 or 1.

## Final numbers (computed by notebook 03)

- **Judged accuracy** (primary) = (correct + 0.5 × partial) / N
- **Strict accuracy** = correct / N
- **Lenient accuracy** = (correct + partial) / N
- Each metric is reported overall, for past, and for future.
- **Comparison:** the difference in judged accuracy in points, with a 95% paired bootstrap CI. Relative gain is also reported. McNemar's exact test is run on strict correctness.
- **Secondary string metrics:** ROUGE-L F1 and chrF against the gold answer.
- **Judge self-consistency:** agreement on the hidden duplicates. This is not an independent check: the judge may remember its first rating.
- **Automatic scores (v2):** abstained = caught by the pattern or flagged by the judge. Answerable item answered with an abstention = 0; "Bilmiyorum." item = 2 if the model abstained, else 0. **Overall score** = mean of score / 2 over all items. Abstention precision, recall and F1 are reported with a cue-only reference (abstain on hangi / kaç / ne kadar questions).

The "+10%" target is reported both ways, as +10 points absolute and as +10% relative. State in the report which one the advisor means.

## Prompt for an API judge (optional)

If a model is used through an API instead of in chat, send each row with this prompt, at temperature 0:

```
Sen bir değerlendiricisin. Aşağıda kısa bir hikâye (Bölüm 1, 2, 3), bir soru, referans cevap ve bir aday cevap var.
Aday cevabı yazan model yalnızca Bölüm 2'yi gördü. Soru yönü: {direction} (past = cevap Bölüm 1'de, future = cevap Bölüm 3'te).
Puanla: 2 = doğru (referansla ya da gizli bölümle aynı temel bilgi), 1 = kısmen doğru (uyumlu ama eksik/daha genel),
0 = yanlış (çelişiyor, farklı bilgi, hikâyede dayanağı olmayan tahmin, genel/boş cevap).
Referans cevap "Bilmiyorum." ise hikâye cevabı vermiyor: aday, herhangi bir ifadeyle bilginin verilmediğini söylüyorsa 2, somut bir cevap veriyorsa 0.
abstains: aday cevabın ana mesajı "bu bilgi verilmemiş / bilinemez" ise 1, somut bir cevap veriyorsa 0 (doğru olup olmamasından bağımsız).
Sadece JSON döndür: {"score": 0|1|2, "abstains": 0|1, "reason": "<10 kelime"}

Bölüm 1: {part1}
Bölüm 2: {part2}
Bölüm 3: {part3}
Soru: {question}
Referans cevap: {gold_answer}
Aday cevap: {candidate_answer}
```
