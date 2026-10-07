# Mock Quiz 4: Assignment 4 (Distributed Training)

> 15 minutes per set. Closed book, no devices. 3 points per set.
> Answer as you would in a design interview: two to four sentences per part, with the reasoning.
> Every number you need is on this sheet. Do **Set A** first and check the answers, then do **Set B**.

---

## Set A

**The setting.** Same team as before. The screenwriting app now needs a bigger model, so you train a
**1.0B-parameter** GPT with your Assignment 4 trainer on one `gpuA40x4` node (four A40s, 44 GiB each).
Weights, gradients and both Adam moments are fp32. Compute runs in bf16 autocast.
The global batch is 128 sequences and the micro-batch is 4.
On one GPU, **backward is about 2/3 of the compute time**.

**All-reduce bus bandwidth, four A40s in one node**

| buffer | 1 MB | 16 MB | 256 MB | 1 GB |
|---|---|---|---|---|
| busbw (GB/s) | 6 | 13 | 16 | 16 |

**Runs**

| run | GPUs | step time (s) | peak memory (GiB) |
|---|---|---|---|
| single GPU | 1 | 8.0 | 20 |
| DP, naive | 4 | 2.5 | 20 |
| DP, overlapped | 4 | 2.2 | 20 |
| TP = 4 | 4 | 9.1 | 7 |
| PP = 4, m = 4, 1F1B | 4 | 3.6 | 12 |
| PP = 4, m = 16, AFAB | 4 | 2.6 | 18 |
| PP = 4, m = 16, 1F1B | 4 | 2.6 | 7 |

### 1. Data parallel (1.5 points)

**(a)** Perfect 4× scaling would give a 2.0 s step, but naive DP takes 2.5 s.
How much of that 0.5 s gap does a ring all-reduce explain? Show the bytes and the time.
What could account for the rest?

**(b)** The cluster team moves you to **four nodes with one A40 each**. Between nodes, the bus bandwidth is
only **4 GB/s**. Predict the naive step time. Your teammate says, "Overlapped DDP will hide most of that."
Will it? Say what sets the ceiling, and use the numbers.

### 2. Tensor vs. pipeline (1.5 points)

**(a)** Product sees TP = 4 at 9.1 s, which is **slower than one GPU**, and asks whether your TP code is broken.
TP = 4 and DP = 4 do the same arithmetic on the same four GPUs. Explain the gap.
Is there any reason to keep TP? Which mapping would you ship for this model?

**(b)** Predict the pipeline bubble for m = 4 and m = 16, and check whether the step times agree with it.
Then explain why AFAB and 1F1B have the same step time at m = 16 but use 18 GiB vs. 7 GiB.

---

## Set B

**The setting.** The team now wants a **4.0B-parameter** model. You have **two `gpuA40x4` nodes**
(8 A40s, 44 GiB each). The global batch is 128 and the micro-batch is 1.
The default axis order places TP innermost (within a node) and PP outermost.

| DP | TP | PP | step time (s) | peak memory (GiB) |
|---|---|---|---|---|
| 8 | 1 | 1 | OOM | — |
| 4 | 2 | 1 | 12.8 | 38 |
| 2 | 4 | 1 | 17.5 | 24 |
| 2 | 1 | 4 | 23.1 | 28 |
| 2 | 2 | 2 | 40.8 | 23 |
| 1 | 8 | 1 | 96.0 | 9 |

### 1. Fitting the model (1.5 points)

**(a)** Why does DP = 8 run out of memory while DP = 4 × TP = 2 fits?
Give the per-GPU memory for model state in each case. What does each parallel dimension buy you in memory?

**(b)** TP = 8 is the most memory-efficient row but the slowest by far: 96 s, versus 17.5 s at TP = 4.
Explain what changed between TP = 4 and TP = 8. State the placement rule you would follow.

### 2. Alternatives and anomalies (1.5 points)

**(a)** A teammate wants to keep pure DP = 8 and proposes **activation checkpointing** to make it fit.
Will that work? If not, name a technique from lecture that would make DP = 8 fit, and say what it costs.

**(b)** In the Part 3 sweep (`large`, p = 4, global batch 32), the **measured idle fraction is always higher
than (p−1)/(m+p−1)**: 54% vs. 43% at m = 4, and 28% vs. 9% at m = 32. Also, as m goes from 4 to 32,
**1F1B's peak memory falls** from 17.1 to 6.7 GiB, while AFAB's rises and runs out of memory at m = 32.
Explain both observations.

---

## Lightning round (no points, answer in one sentence each)

1. `f` is an identity in the forward pass. What breaks if it is also an identity in the backward pass?
2. Why does `RowParallelLinear` add its bias **after** the all-reduce?
3. In `OverlappedDDP`, a teammate moves `p.grad /= world` to just **after** launching the async all-reduce.
   Tests pass sometimes and fail sometimes. Why?
4. Why must the steady state of 1F1B use `send_forward_recv_backward`, rather than a blocking send followed by a blocking recv?
5. Why does `broadcast_module_state` sync `buffers()` as well as `parameters()`?
6. Why is the PP stage boundary, not TP, the axis that should cross the slow link between two nodes?
7. With gradient accumulation, why does `no_sync()` matter for communication volume?
