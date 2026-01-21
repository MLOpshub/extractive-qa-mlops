# Idea Log — RAG / Training Focus (English Version)

> Purpose: Record the instructor’s email and our reflections on the project direction for continuous updates later.  
> Status: **Draft** (to be discussed with teammates)

---

## 1 Background
Our current project is primarily **Retriever–Reader (FAISS + Extractive QA Reader)**. The course emphasizes **training** and core MLOps capabilities (reproducibility, evaluation, tracking, deployment, monitoring).

---

## 2 logs

### **2026-01-21 00:36**, Europe/Paris - Email

✉ Original
```
Hi Tengzhe,
Regarding the issue we discussed today, I talked with the company’s founder and have some new thoughts to share with you.

RAG evaluation can be approximately viewed as an ML process. We can use an LLM to modify the prompt (similar to gradient descent) to find an optimal prompt that maximizes RAG performance.
Here, the input is the prompt and the output is the KPIs from evaluation. You can combine KPIs into a cost function for training, treating the entire RAG+LLM process as a single overall algorithm.

Note that in this setup we need to use the LLM three times: once for RAG, once for prompt optimization, and once for evaluation. During training, each iteration triggers all three, which can be very expensive in tokens (money). This is something you need to consider.

This is our first year running the course, so the content may be somewhat conservative. Nowadays, LLMOps is rapidly replacing traditional MLOps, and much ML is also done through LLMs. But we believe the fundamental mindset is unchanged; you can see the similarity to an ML process through this evaluation workflow.

Hope the above helps you.
```

🖊 My understanding of the email
```
The instructor is not emphasizing “you must use an LLM,” but rather:
- Treat the system as an end-to-end algorithm that can be optimized
- Clearly define tunable parameters / control variables (prompt / policies / thresholds / component choices)
- Use offline evaluation KPIs as feedback signals for iterative improvement
- Consider cost constraints (tokens / compute / latency)
```

💡 idea
```
❌ Directions we decide NOT to pursue (to avoid making the project too hard or drifting from requirements)
① Full-pipeline cost function + automated optimization  
  - Optimization spans multiple modules and would require unified orchestration plus a large number of combinatorial experiments, significantly increasing complexity and coordination cost.  
  - This can easily evolve into AutoML / system-level search, beyond our baseline objectives.
② Prompt search (multiple LLM calls)  
  - Our baseline does not rely on an LLM; introducing prompt search would deviate from the course requirement that “training is the focus.”  
  - Token cost and engineering complexity are high.

✔ Optimization mindset we CAN borrow
① Use the “trained Reader” for lightweight post-processing optimization, implemented via configs (e.g., `configs/prompts/*.txt` or `configs/prompts.yaml`), analogous to prompt tuning.  
Tunable parameters:
- top_k: number of retrieved passages (e.g., 3/5/10)  
- weak_retrieval_threshold: trigger fallback when the top retrieval score is too low  
- answer_selection_rule: how to select the final answer from multiple spans across multiple passages  
  - max_score  
  - score + length_penalty (to avoid overly long spans)  
  - confidence_gating (fallback when confidence is too low)
```
