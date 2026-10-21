# gemma-tr-temporal-inference

Dil Modellerinde Bağlamdan Geçmiş-Gelecek Çıkarımı. We fine-tune `google/gemma-3-1b-it` with LoRA so that, given only the middle of a Turkish passage (part 2), it answers a question about what happened before (past) or after (future). We compare the untrained and the fine-tuned model on the same test pairs using judged accuracy.