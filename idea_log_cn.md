# Idea Log — RAG / Training Focus（中文版）

> 用途：记录老师邮件与我们对项目方向的思考，便于后续持续更新。  
> 状态：**Draft**（待与组员讨论）

---

## 1 背景
我们当前项目以 **Retriever–Reader（FAISS + Extractive QA Reader）** 为主，课程强调 **training** 与 MLOps 基本能力（可复现、评估、追踪、部署、监控）。

---

## 2 logs

### **2026-01-21 00:36**，Europe/Paris - 邮件

✉ 原文
``` 
> 你好螣喆，
关于我们今天讨论的问题，我与公司的founder讨论了一下有了些新的想法跟你分享一下。

RAG的evaluation可以近似看为一个ML过程，我们可以通过LLM修改prompt（类似于梯度下降），找到最优的prompt来最大化RAG的performance。
这里的输入是prompt，输出是evaluation中的KPIs，可以组合KPI做成一个cost function进行训练，整个的RAG+LLM过程看成一个整体的algorithm。

需要注意的是，这里我们需要三次用到LLM：RAG一次，prompt优化一次，evaluation一次，而且在training过程中每进行一次iteration都要trigger这三次，这对tokens（钱）的消耗是非常大的，这是你们需要考虑的问题。

今年是我们第一次开课，内容可能稍微有些保守，现在的LLMOps已经在快速取代过去的MLOps，很多ML也是通过LLM进行的。但我们认为基本的mindset是不变的，通过这个evaluation的过程也能看出来其与ML过程的相似性。

希望以上可以帮助你。 
```

🖊 我对邮件的理解
```
老师强调的不是“必须用 LLM”，而是：
- 把系统当成一个 **可优化的算法整体**  
- 明确 **可调参数/控制变量**（prompt / 策略 / 阈值 / 组件选择）  
- 用 **离线评估 KPI** 作为反馈信号，进行迭代改进  
- 同时考虑 **成本约束**（token/compute/latency）
```

💡 idea
```
❌决定不做的方向（避免项目过难、偏离要求）
① 全管线 cost function + 自动优化
  - 优化涉及多个模块，需要统一编排与大量组合实验，会显著增加复杂度与协作成本。
  - 容易演变为 AutoML/系统级搜索，超出 baseline 目标。
② prompt search（多次 LLM 调用）
  - 我们 baseline 并不依赖 LLM；引入 prompt search 会偏离“训练是重点”的项目要求。
  - token 成本与工程复杂度高。

✔ 可借鉴优化思路
① 使用“已训练的 Reader”做轻量级后处理优化，优化进行在configs/下，例如configs/prompts/*.txt 或 configs/prompts.yaml（类比 prompt 优化）
可调参数：
- top_k：检索返回数量（例如 3/5/10）
- weak_retrieval_threshold：当检索 top score 过低时触发 fallback
- answer_selection_rule：如何从多个 passage 的多个 span 中选最终答案
  - max score
  - score + length penalty（避免过长 span）
  - confidence gating（置信度过低时 fallback）



