---
marp: true
theme: default
paginate: true
size: 16:9
---

<style>
section.outlier-slide {
  padding-top: 36px;
}

section.outlier-slide h1 {
  margin-top: 0;
  margin-bottom: 0.2em;
}

section.outlier-slide ul {
  margin-top: 0.2em;
  line-height: 1.18;
}

section.outlier-slide li {
  margin: 0.08em 0;
}

section.two-col-critique .columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.2rem;
  align-items: start;
}

section.two-col-critique h3 {
  margin: 0 0 0.25em 0;
}

section.two-col-critique ul {
  margin-top: 0.1em;
  padding-left: 1.1em;
}

section.compact-related {
  font-size: 28px;
}

section.compact-related h1 {
  margin-bottom: 0.2em;
}

section.compact-related p {
  margin: 0.2em 0 0.05em 0;
}

section.compact-related ul {
  margin-top: 0.05em;
  margin-bottom: 0.25em;
  line-height: 1.12;
}

section.compact-related li {
  margin: 0.04em 0;
}

section.title-slide footer {
  position: absolute;
  color: #9aa0a6;
  font-size: 18px;
  left: 6%;
  right: auto;
  bottom: 10% !important;
}
</style>

<!-- _class: title-slide -->
<!-- _footer: Kartik Ramesh -->
# WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training
*Zheng Wang, Anna Cai, Xinfeng Xie, Zaifeng Pan, Yue Guan, Weiwei Chu, Jie Wang, Shikai Li, Jianyu Huang, Chris Cai, Yuchen Hao, Yufei Ding*

---
<!-- _footer: "" -->

# Introduction
- Llama 3.1 405B trained over 16k GPUs for 30M H100 hours.
- Using AWS H100 pricing **$212M**.
- A 1.2x increase in training speed would have saved **$36M** in cloud costs.
- WLLB-LLM is a work from Meta and UCSD, released ~6 months after their training Llama 3 that achieves this.

---
# Self Attention
- Attention scales quadratically with the number of past tokens $O(T^2)$
- Not all tokens are equal: Processing later tokens in a document require more computation than earlier.
- When packing multiple documents for training, control attention span using masks.
![bg right:45% contain](attachments/attention_masks.png)

---
# 4D Parallelism
- Huge models are trained on huge clusters because of scale and necessity.
- Tensor shape $[B, T, H]$ across several layers.
- 4D Parallelism splits this shape in various ways.
    - Data - Split $B$ across replicated model.
    - Pipeline - Split model into groups of layers.
    - Context - Split along sequence length $T$
    - Tensor - Split across the $H$.

---
# 4D Parallelism
<div style="height:68%; display:flex; align-items:center; justify-content:center;">
  <img src="attachments/4d_parallelism.png" style="max-width:100%; max-height:100%; object-fit:contain;" />
</div>

$$
DP > PP > CP > TP
$$

---

# Key Takeaways
**Problem**: In long-context 4D training, token count is a weak proxy for compute; attention cost is highly non-uniform.
- **Core idea #1 (PP)**: Reduce PP imbalance via attention-aware micro-batch packing.
- **Core idea #2 (CP)**: Reduce CP imbalance via fine-grained per-document sharding.

**Bottom line**: WLB-LLM improves training throughput ($\approx 1.23x$) without hurting convergence.

---
# Motivation #1 - Pipeline Imbalance
- Pipeline Parallelism splits $B$ into $N$ $\mu$B to hide delays.
- Balanced Batches $\rightarrow$ Higher throughput

![bg right:40% contain](attachments/pp-bubbles.png)

---

# Motivation #1 - Pipeline Imbalance
- Naive approach $\rightarrow$ split by tokens.
- Each $\mu$B has uniform sequence length = $\texttt{CONTEXT\_WINDOW\_SIZE}$
- Does this balance LLM workload?
- $k.128^2 >> k.128 * 1^2$

![bg right:40% contain](attachments/pp-docs-1.png)

---
# Motivation #2 - Context Imbalance
- Workload across CP workers should be balanced.
- After packing, split the sequence into $2.CP$ parts and assign one from front and one from back to balance attention.
- Good heuristic, fails for multiple packed documents. Common in long context training.
- Every small delay adds up to higher-order delays.
![bg right:30% contain](attachments/cp-imbalance.png)

<!-- ---
  - show figure 3
      - 3a motivates that there is a lot of variance in the input document size. long inputs can cause workload imbalance.
      - 3b motivates that absolute token position is actually a bad proxy for computation intensity.
          - perhaps a better metric might have been average in-document prefix length, since that is the one that matters.
          - look at their presentation, if available to see how they argue. -->
---
# Baseline: Attention-Aware packing.
**Idea**: Divide $B$ into $\mu$B by estimating $d_i^2$ as attention cost for each document.

- It works, but limited balancing improvements $\rightarrow$ limited speedup.
- Higher balancing across $\mu$B requires balancing across multiple global $B$. This disturbs the random order of training and loss convergence.
- It might be impossible to come up with such a $\mu$B construction, if there are no candidates.

![bg right:40% contain](attachments/baseline-algo.png)

---

# Variable-Length Packing
**Idea**: Allow $len(\mu B) > \texttt{CONTEXT\_SIZE}$ for weaker $\mu B$

- Attention is Quadratic, but other operations are linear (feed-forward, comms etc.)
- Balance the total workload, not just attention.
- Balance long documents against many shorter documents.

$\min \left( \max \left( \sum_{i=1}^{N} \big( W_a(x_{ij} \cdot d_i) + W_l(x_{ij} \cdot d_i) \big) \right) \right)$

![bg right:40% contain](attachments/attention-v-linear.png)

---

<!-- _class: outlier-slide -->
# Outlier document detection
- Still, you might not have sufficient smaller documents to balance the load of a long document.
- Observe: there aren't that many ultra-long documents.
- Instead of balancing across multiple batches, delay the few long documents.
![bg right:25% contain](attachments/docs-length.png)
- Model convergence should not hurt significantly.
<!-- - Key observation: such long documents are not that frequent. Let's delay them.
    - Suppose you are trying to form N micro-batches on every iteration.
    - Use a single queue, that holds documents that are greater than a threshold L1. (outlier)
    - When you get a global batch, any outlier documents (length > L1) are moved into this queue.
    - When this queue reaches a size N, pop each document and spread it across the N micro-batches.
    - This ensures no single micro-batch has an outlier.
    - Extend this to a multi-queue with various outlier lengths: L1, L2, .. , Ln. -->
<!-- - Figure 8. highlights this well. -->

<div style="position:absolute; left:3%; right:3%; bottom:5%; height:9%; display:flex; justify-content:center; align-items:center;">
  <img src="attachments/outlier-queue.png" style="display:block; width:100%; height:100%; object-fit:contain;" />
</div>

<!-- ---
# Main Algorithm
**TODO @kartik fix the presentation of this.**
Given a DP batch
- Construct a document set by removing the outlier documents, popping from the queue, and rolling over any documents from the previous iteration.
- now you effectively have a set of documents that you solved for using an ILP in S3.3, however they claim that it would take too long.
- so they instead pack greedily into an array of N micro-batches. sort the documents in reverse by length, and for each document:
    - find the least loaded microbatch (first by load, next by length)
    - if it can fit this current document, great.
    - else reserve this document for a later stage. -->

<!-- ![bg contain](attachments/main-algo.png) -->

---

# Improved CP sharding
**Idea**: Apply CP indexing logic to each individual document.
- This should yield a more balanced workload across multiple CP workers.
- They also implement an optimization to avoid padding tokens.

<div style="position:absolute; left:6%; right:6%; bottom:5%; height:36%; display:flex; justify-content:center; align-items:center;">
  <img src="attachments/CP-indexing.png" style="display:block; width:100%; height:100%; object-fit:contain;" />
</div>

---

# Kernel inefficiencies
- Per-Document sharding achieves better balance, but it does not always guarantee better performance.
- Smaller per-rank attention problems reduce kernel efficiency:
  - Poor tile utilization → padding overhead for short sequences (<128 tokens).
  - Lower effective FLOPs utilization → higher time per token.
  - Reduced KV tile reuse → weaker Hopper TMA multicast benefits

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

---
# Kernel Inefficiencies

<div style="position:absolute; left:6%; right:6%; top:18%; bottom:8%; display:flex; justify-content:center; align-items:center;">
  <img src="attachments/kernels.png" style="max-width:100%; max-height:100%; object-fit:contain;" />
</div>

---

# Experimental Setup
- **Cluster**: 32 nodes, each with 8x NVIDIA H100 SXM 80GB GPUs.
- **Interconnect**: NVLink intra-node, RoCE inter-node.
- **Models**: LLaMA-like 550M, 7B, 30B, 70B; each tested at 64K and 128K context.
- **Training config**: 4D parallelism, global batch size = `PP_size x DP_size`, `bfloat16` precision.
- **Baselines**:
  - `Plain-4D`: default 4D training with per-sequence CP sharding.
  - `Fixed-4D`: fixed-length packing + fixed CP sharding (per-sequence or per-document).

---

# Speedup Breakdown
![bg right:45% contain](attachments/fig-13.png)
<!-- Kartik note: annotate bars to highlight PP var-len + outlier delay as the largest contributor (1.28x), and final combined gain (~1.33x). -->

Which optimization helps us the most?

- $\texttt{PP-Var-Len}$ alone $\rightarrow$ 1.28x
- Orthogonal optimizations that combine well.
- Every second counts!

---

# Speedup across Model + Context

<div style="position:absolute; left:5%; right:5%; top:18%; bottom:30%; display:flex; justify-content:center; align-items:center;">
  <img src="attachments/fig-12.png" style="max-width:100%; max-height:100%; object-fit:contain;" />
</div>
<!-- Kartik note: annotate the two main trends: larger context => larger gains; larger models => slightly smaller relative gains due to communication overhead. -->

- WLB-LLM consistently outperforms baseline for all tested configurations.
- Naive attention balancing is insufficient.
- Relative speedup decreases with increased model size.

---
# Other Experiments (Summary)
- **Context sensitivity (Fig. 14)**: Speedup increases with context length (about `1.07x @64K` to `1.40x @160K` on 7B), consistent with worse imbalance at longer contexts.
- **Packing overhead vs balance (Table 2)**: WLB-LLM reaches near-optimal imbalance with low runtime overhead (tens of ms), while solver-based packing can be prohibitively slow.
- **CP sharding ablation (Fig. 15)**: Adaptive sharding consistently outperforms always-per-sequence or always-per-document sharding.
- **Convergence / quality**: Their training-loss curves indicate no clear quality regression from the system optimizations.

---
<!-- _class: two-col-critique -->
# Discussion & Critique
<div class="columns">
  <div>
    <h3>Strengths</h3>
    <ul>
      <li>Identifies and fixes a bottleneck for training long-context Llama models with 4D parallelism.</li>
      <li>Joint PP+CP optimization gives meaningful real-system gains, especially as context windows grow.</li>
      <li>Engineering is practical: low overhead and no obvious convergence regression in their reported runs.</li>
    </ul>
  </div>
  <div>
    <h3>Weaknesses</h3>
    <ul>
      <li>Workload dependence and limited generalization evidence outside their evaluated distribution.</li>
      <li>Strong dependence on a small number of extreme outliers; unclear benefit when length distributions are flatter.</li>
      <li>Heavy use of heuristics (packing + sharding selection) without strong guarantees in worst-case settings.</li>
    </ul>
  </div>
</div>

---
<!-- _class: compact-related -->
# Related Work
**(1) Efficient Long-context Language Model Training by Core Attention Disaggregation (DistCA)**
- Split out “core attention” ($\mathrm{softmax}(QK^\top)V$) as a weightless compute service, separate from the rest of the transformer.
- Better than WLB-style baselines at scale: reports ~1.15–1.35× throughput gains over their WLB “ideal” baseline in 4D (with PP), depending on workload.


**(2) ByteScale Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs**
- Hybrid Data Parallelism (HDP): unify DP + CP into one dynamic device mesh. 
* Length-aware sharding: use the minimum number of devices per sequence.
* Short sequences stay local (skip CP comm), long sequences shard across more GPUs. 

**(3) Ordering efficiency**
- Reduce pipeline bubbles by optimizing scheduling.
- PipeDream, 1F1B, Seq1F1B.

---
# What did you think?
**Problem**: In long-context 4D training, token count is a weak proxy for compute; attention cost is highly non-uniform.
- **Core idea #1 (PP)**: Reduce PP imbalance via attention-aware micro-batch packing.
- **Core idea #2 (CP)**: Reduce CP imbalance via fine-grained per-document sharding.

**Bottom line**: WLB-LLM improves training throughput ($\approx 1.23x$) without hurting convergence.

---
![bg contain](attachments/table-2.png)


<!-- --- -->
<!-- # Evaluation: Other Results
- table 2.
    - their wlb solver achieves a low degree of imbalance, with low overhead.
    - important to note how they define imbalance.
- little performance degradation.
- figure 15.
    - speedups for different cp strategies, for constant context length.

---
# general presentation notes
- include piazza questions naturally in the flow of the presentation if highly aligned.
- include results naturally in the flow if it helps. -->
