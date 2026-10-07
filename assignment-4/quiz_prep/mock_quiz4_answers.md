# Mock Quiz 4: Answer Key

每一題都有兩個部分：**得分要點**模仿 Gradescope 的評分方式，**範例答案**是考試時可以直接寫的英文，每題 2 到 4 句。

---

## Set A

### 1(a) Ring all-reduce vs. the 0.5 s gap

**得分要點**
- Bytes: 1.0B × 4 B = **4 GB** of fp32 gradients. A ring moves 2(P−1)/P = **1.5×** that per rank, so **6 GB**.
- Time: 6 GB ÷ 16 GB/s = **0.375 s**, which explains **75%** of the 0.5 s gap.
- The rest: the reduction is hundreds of **per-parameter all-reduces**. Small tensors run at the latency-bound end (6 GB/s at 1 MB), and each call also has launch and synchronization overhead.

**範例答案**
> Ideal is 8.0/4 = 2.0 s, so naive pays 0.5 s of overhead. The gradients are 1.0B × 4 B = 4 GB, and a ring all-reduce puts 1.5 × 4 = 6 GB on each link; at 16 GB/s that is 0.375 s, about 75% of the gap. The remaining ~0.12 s comes from issuing one all-reduce per parameter: small buffers are latency-bound (6 GB/s at 1 MB), and each call adds launch and sync overhead.

### 1(b) Moving to four nodes at 4 GB/s

**得分要點**
- Communication: 1.5 × 4 GB ÷ 4 GB/s = **1.5 s**, so naive ≈ 2.0 + 1.5 ≈ **3.5 s**.
- Overlap **cannot** hide most of it. With accumulation, `no_sync` means only the **last micro-batch's backward** launches reductions. Per rank there are 128 ÷ (4 × 4) = **8** micro-batches, so that backward is about 2.0 s ÷ 8 × 2/3 ≈ **0.17 s**. That is the most overlap can hide.
- Also, the first layers' gradients arrive last, so their reduction is always exposed. Expect a step of **≥ 3.3 s**, which is only about 60% scaling efficiency.

**範例答案**
> At 4 GB/s the same 6 GB takes 1.5 s, so naive goes to about 3.5 s. Overlap can only hide communication behind backward compute that is still running, and with gradient accumulation only the last micro-batch's backward triggers reductions: 8 micro-batches per rank, so roughly 0.25 s × 2/3 ≈ 0.17 s of cover. So overlap saves at most ~0.17 s of the 1.5 s, and the step stays above ~3.3 s; the ceiling is the backward time available to overlap, not the code.

### 2(a) TP = 4 is slower than one GPU

**得分要點**
- Same FLOPs, so the gap of 9.1 − 2.2 ≈ **7 s** is **exposed communication**.
- TP all-reduces **activations**, 4 times per block per micro-batch (2 in forward via `f_bar`, 2 in backward via `f`). They sit **on the critical path**: the next layer waits for the result, so nothing hides them. DP all-reduces the gradients **once per step**, and that overlaps with backward.
- What TP buys is **memory**: 7 GiB vs. 20 GiB, because weights, gradients and Adam state are split four ways.
- Verdict: the model fits (20 GiB < 44 GiB), so **ship DP = 4**. Use TP only when the model does not fit.

**範例答案**
> TP=4 and DP=4 do the same arithmetic, so the ~7 s gap is communication that cannot be hidden: TP all-reduces activations four times per block for every micro-batch, and each one blocks the next layer, whereas DP reduces gradients once per step and overlaps it with backward. The code is not broken; TP trades time for memory, and its 7 GiB vs. 20 GiB is the benefit. Since this model fits on one A40 at 20 GiB, I would ship DP=4 and only reach for TP when the model state no longer fits.

### 2(b) Bubble prediction and AFAB vs. 1F1B memory

**得分要點**
- Bubble = (p−1)/(m+p−1): m = 4 → 3/7 ≈ **43%**. m = 16 → 3/19 ≈ **16%**.
- With no bubble, each stage would take 8.0/4 = **2.0 s**. Predicted step = 2.0 ÷ (1 − bubble):
  - m = 4: 2.0 × 7/4 = **3.5 s**, vs. 3.6 s measured ✓
  - m = 16: 2.0 × 19/16 ≈ **2.4 s**, vs. 2.6 s measured. The extra ~0.2 s comes from **P2P transfers** and **unequal stages** (the last stage also carries the LM head and the loss).
- The bubble depends only on p and m, not on the order, so **the step time is the same** for both schedules.
- Memory: **AFAB** keeps all m = 16 micro-batches' activations alive until backward starts. **1F1B** keeps **at most p = 4** in flight. That is 18 GiB vs. 7 GiB.

**範例答案**
> The bubble is (p−1)/(m+p−1): 3/7 ≈ 43% at m=4 and 3/19 ≈ 16% at m=16. A bubble-free stage would take 8.0/4 = 2.0 s, giving 2.0 × 7/4 = 3.5 s and 2.0 × 19/16 ≈ 2.4 s, which match 3.6 s and 2.6 s up to P2P transfers and the heavier last stage. AFAB and 1F1B have the same bubble because it depends only on p and m, but AFAB holds all 16 micro-batches' activations until backward begins, while 1F1B never holds more than p = 4, hence 18 vs. 7 GiB.

---

## Set B

### 1(a) Why DP = 8 runs out of memory but DP4 × TP2 fits

**得分要點**
- Model state = 4.0B × **16 B** (4 for weights, 4 for gradients, 8 for Adam) = **64 GB** (≈ 60 GiB).
- **DP replicates the model state** and saves no memory, so every GPU in DP = 8 needs 64 GB, which is more than 44 GiB. It runs out of memory before any activations.
- **TP and PP divide the model state by tp·pp.** DP4 × TP2 needs 64/2 = **32 GB**, plus activations, which matches the 38 GiB measured.
- What each dimension buys: **DP** buys throughput but no memory. **TP** splits each layer's weights and part of the activations. **PP** splits the layers, but each stage must keep up to p micro-batches of activations.

**範例答案**
> The model state is 4.0B × 16 bytes = 64 GB, and DP replicates it, so each DP=8 rank needs 64 GB before activations, which exceeds a 44 GiB A40. TP and PP divide the state by tp·pp: with TP=2 each GPU holds 32 GB, and activations bring it to the 38 GiB measured. DP only buys throughput; TP and PP are the dimensions that buy memory.

### 1(b) Why TP = 8 is so much slower than TP = 4

**得分要點**
- With 4 GPUs per node, a **TP = 4 group fits inside one node**, but a **TP = 8 group spans both nodes**.
- Each block's activation all-reduces, which sit on the critical path, now cross the **slow inter-node link**. They also run as an 8-way ring.
- Placement rule: **keep TP inside a node** (highest bandwidth, most frequent communication). Put **PP across nodes**, since it sends only one activation per micro-batch at a stage boundary. DP can also cross nodes, because it communicates once per step and overlaps with backward.

**範例答案**
> With four GPUs per node, a TP=4 group stays inside one node, but TP=8 spans both, so every block's activation all-reduces, which are on the critical path, now travel over the much slower inter-node link. That turns the most frequent collective into the slowest one, hence 96 s vs. 17.5 s despite the lowest memory. The rule is to keep TP within a node and let PP (one point-to-point activation per micro-batch) or DP (once per step, overlappable) cross nodes.

### 2(a) Activation checkpointing vs. ZeRO

**得分要點**
- **It will not work.** Checkpointing only shrinks **activations**, and the 64 GB of model state is already larger than 44 GiB.
- **ZeRO** removes the redundancy across DP ranks:
  - ZeRO-1 splits Adam's 32 GB across 8 GPUs (4 GB each), giving 16 + 16 + 4 = **36 GB**. That is tight once activations are added.
  - ZeRO-2 also splits the gradients: 16 + 2 + 4 = **22 GB**.
  - ZeRO-3 splits everything: 64/8 = **8 GB**.
- Cost: **more communication**. Gradients need a reduce-scatter, and parameters need an all-gather; under ZeRO-3 that happens in every forward and backward pass. All of it would cross the inter-node link.

**範例答案**
> No: activation checkpointing trades recompute for activation memory, but the 64 GB of weights, gradients and Adam state alone already exceed 44 GiB. ZeRO removes DP's redundancy instead: ZeRO-1 shards the 32 GB of Adam state 8 ways (→ ~36 GB per GPU), ZeRO-2 also shards gradients (→ ~22 GB), and ZeRO-3 shards parameters too (→ ~8 GB). The cost is extra communication, reduce-scatter for gradients and all-gather for parameters, every step and across the inter-node link.

### 2(b) Idle above the formula, and 1F1B memory falling

**得分要點**
- **The measured idle fraction is higher than predicted** because the formula assumes **equal stage times and free communication**. In reality:
  - Stages exchange activations and gradients **point to point**, and each stage waits for those transfers.
  - Stages are **unequal**: the last stage carries the 50304-wide LM head and the loss, so the other stages wait for it.
  - These costs are paid **per micro-batch**, so the gap does not disappear as m grows.
- **1F1B memory falls** because the global batch is fixed at 32, so micro = 32/m keeps shrinking. 1F1B holds at most p micro-batches, and p × micro = 4 × 32/m gets smaller.
- **AFAB** holds all m micro-batches. The total m × micro = 32 stays the same, so in theory its memory should be flat. It rises because of **overhead per micro-batch** (saved tensors and allocator fragmentation) on a last stage that already holds the big logits. At m = 32 it goes over 44 GiB.

**範例答案**
> The formula assumes identical stages and free communication; in practice each micro-batch pays point-to-point activation and gradient transfers, and the last stage carries the 50304-wide LM head and loss, so the other stages wait on it. Those costs scale with m, so idle stays above the prediction. 1F1B memory falls because the global batch is fixed: micro = 32/m shrinks while 1F1B holds at most p micro-batches, whereas AFAB holds all m (constant m×micro in theory, plus per-micro-batch overhead on the logits-heavy last stage), which pushes it past 44 GiB at m = 32.

---

## Lightning round

1. **`f` is the identity in both directions**: each rank sees only its shard's part of the input gradient. LayerNorm, the earlier blocks and the embedding all get **wrong, partial gradients**, and the replicated weights drift apart across ranks.
2. **Bias added before the reduction**: every rank adds it, so after the sum it has been added **tp times**.
3. **Dividing after launching the async all-reduce**: this is a **race**. The all-reduce is still reading and writing `p.grad`, so the result depends on timing and changes from run to run. Divide before launching, or after `wait()`.
4. **Blocking send, then blocking recv**: two neighbours can both block in `send`, each waiting for the other to `recv`, and **deadlock**. The batched `isend`/`irecv` posts both at once.
5. **Syncing `buffers()`**: buffers such as masks or BatchNorm running statistics change the forward pass. If they differ across ranks, the outputs differ even when the parameters match.
6. **PP across nodes**: a PP boundary sends **one point-to-point activation per micro-batch**. TP issues several all-reduces per block on the critical path.
7. **`no_sync()`**: without it, **every micro-batch** all-reduces the full gradient, which is accum× the traffic (for example, 8 × 4 GB). With it, only the last micro-batch does.
