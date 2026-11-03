# Mock Quiz 6: Answers and Rubric

Each part is worth 0.75 points. The rubric follows the real quizzes: **mechanism + numbers + a concrete
verdict**. A correct verdict without the numbers, or numbers without the reason, gets partial credit.

---

## Set A

### 1(a) Sizing the cache

**Answer.**
KV per token = 2 (K, V) × 36 layers × 1280 × 2 B = **184,320 B (180 KiB)**.
- **Contiguous:** each request reserves 1,024 slots = 184,320 × 1,024 = 188.7 MB.
  20 × 10^9 / 188.7 × 10^6 = 106.0, so **105 requests**.
- **Paged, block 16:** one block = 16 × 184,320 = 2,949,120 B, so the pool has
  20 × 10^9 / 2,949,120 = **6,781 blocks**. A request needs at most 70 + 40 = 110 tokens = 7 blocks.
  6,781 / 7 = **968 requests**, about **9.2×** more.

**Rubric.**
- 0.25: KV bytes per token with the formula (2 × layers × width × bytes)
- 0.25: contiguous count from 1,024 reserved slots
- 0.25: paged count from the request's real length, rounded up to whole blocks

**Common mistakes.** Forgetting the factor 2 for K and V; using 4 bytes (fp32) instead of 2 (bf16);
reserving only 110 slots for the contiguous cache (it must reserve the whole context, because it does not
know the output length in advance).

### 1(b) The SLO decides max_batch

**Answer.**
The largest batch whose step fits in 30 ms is **B = 256** (24.5 ms; B = 512 takes 41 ms). That gives
256 / 0.0245 s ≈ **10,400 tokens/s**. The contiguous cache can only run 105 requests: between B = 64 and
128, the step is about 16 ms, so about 105 / 0.016 ≈ **6,400 tokens/s**. Paging buys about **1.6×**, not 9×.

Response to product: paging removes the **memory** limit, but throughput stops growing once another limit
binds. Up to about B = 128, decode is memory-bound (the step time barely moves, 14.0 → 17.0 ms), so a
bigger batch is almost free. Past that, the step becomes compute-bound and the step time rises with the
batch, and at B = 256 the **TBT SLO** is the limit. The extra memory (968 − 256 requests' worth) can be used
for something else, such as prefix caching or longer contexts.

**Rubric.**
- 0.25: picks B = 256 from the table and the SLO
- 0.25: tokens/s for both caches (about 10.4K vs. about 6.4K)
- 0.25: explains why it is not 9×: memory-bound → compute-bound, and the SLO now binds, not memory

### 2(a) Keep optimistic admission

**Answer.** Disagree. Optimistic admission raises throughput from 2,780 to 3,350 tokens/s (+20%) and the
mean batch from 45.2 to 62.5 (+38%), and it lowers both mean latency (2.9 → 2.4 s) and p90 (4.9 → 4.2 s).
The safe engine reserves 7 blocks for every request's full length, so at 1 GB (339 blocks) it can hold at
most 48 requests, and most of those reserved blocks are still empty. Optimistic admission reserves only the
next token, so more requests run per step.

A preemption costs less than it looks because recomputation is **one prefill** of the prompt plus the
tokens generated so far. Prefill processes all those tokens in one forward pass and uses the GPU's compute
well. The original tokens took many memory-bound decode steps. The work is not repeated token by token.

Keep it on, but watch the worst case: a request preempted several times has a much worse latency than the
mean shows, and with an even smaller pool, preemptions could repeat (thrashing) and throughput could fall
below safe admission.

**Rubric.**
- 0.25: verdict with numbers (tok/s and mean batch both up, latency down)
- 0.25: why recompute is cheap (one compute-efficient prefill vs. many memory-bound decode steps)
- 0.25: names the risk (tail latency of preempted requests, or thrashing at smaller budgets)

### 2(b) The short-prompt bug

**Answer.** The **mask** is most likely wrong: it does not hide each row's padded slots. A short row's
block table is padded on the right with **block id 0**, and block 0 is a real block that another sequence
may be using. The gather copies it into the short row's K and V. The mask must allow slot `s` only when
`s <= positions[b, t]`, the row's **own** position. If the new hire builds the mask from the longest row's
positions and shares it with every row (for example `positions.max(0)`), the short rows attend to another
sequence's keys and values.

It only shows up for shorter prompts because the longest row has no padding, so every slot it sees is its
own. With one sequence, or prompts of the same length, no row is padded, so the bug is hidden.

**Rubric.**
- 0.25: names the mask (or the padded table entries) as the cause
- 0.25: explains that padded entries point to block 0, which holds another sequence's data
- 0.25: explains why the longest row and equal-length batches are not affected

**Other acceptable answer (partial credit).** A wrong `slots` computation would also corrupt the cache,
but it would usually break single-sequence tests too, so the mask is the better diagnosis.

---

## Set B

### 3(a) Prefix caching and the timestamp

**Answer.** Automatic prefix caching hashes **full 16-token blocks**, and each hash includes its parent's
hash. The fixed instruction is 50 tokens, so 3 full blocks (**48 tokens**) are identical in every request.
The 4th block mixes the last 2 instruction tokens with the attribute list, which differs per listing.
So 48 of 70 prompt tokens (**69%**) can be reused.

The timestamp breaks this because a token's K and V depend on **every earlier token and its position**. A
different first token changes the first block's hash, and every later hash includes it, so no block matches
and the hit rate drops to 0%. Fix: put anything that changes per request (timestamps, user IDs) at the
**end** of the prompt, or log it outside the prompt.

**Rubric.**
- 0.25: 48 tokens / 69%, with the full-block reason
- 0.25: why a different first token breaks every later block (K, V depend on the whole prefix; chained
  hash)
- 0.25: the fix (move changing content to the end)

### 3(b) Chunked prefill

**Answer.** **Chunked prefill with stall-free batching** (Sarathi-Serve). Each step has a token budget τ:
first add every running decode (100 tokens), then fill the rest of the budget with a chunk of the summary's
prompt. The largest τ whose step fits in 30 ms is **τ = 512** (27 ms; 768 tokens take 38 ms).

Each step then has room for 512 − 100 = 412 prefill tokens, so the 1,000-token prompt takes 3 steps
(412 + 412 + 176). Today, the one big step holds 1,100 tokens, slower than 50 ms, so every autocomplete user
waits more than 3× a normal 15 ms step. With chunks, no step goes over 27 ms.

The summary request pays with a **higher TTFT**: its prefill now spans 3 steps instead of one. The last
step holds 176 + 100 = 276 tokens, so it takes about 18 ms, for about
27 + 27 + 18 = 72 ms in total. Its later chunks also re-read the KV of the earlier chunks, and each extra step repeats the
fixed per-step cost.

**Rubric.**
- 0.25: names chunked prefill / stall-free batching and why it works (decode steps are memory-bound, so
  there is spare compute for prefill tokens)
- 0.25: τ = 512 from the table and the SLO, with the number of chunks
- 0.25: the cost to the summary request (higher TTFT, KV re-reads, repeated per-step overhead)

### 4(a) GQA and capacity

**Answer.** KV per token = 2 × 32 layers × 8 KV heads × 128 × 2 B = **131,072 B (128 KiB)**.
One 8K conversation = 131,072 × 8,192 = 1.07 GB. The KV budget is 80 − 16 − 4 = 60 GB, so
60 / 1.07 = 55.9, and **55 conversations** fit.

With full multi-head attention (32 KV heads), each token needs **4×** as much, 512 KiB, so one conversation
needs 4.29 GB, and only 60 / 4.29 = 13.97, or **13 conversations**, fit. GQA shares one K/V head across 4
query heads, which is why Llama-3.1-8B uses it.

**Rubric.**
- 0.25: KV per token with `n_kv_heads` (not `n_heads`)
- 0.25: the budget (60 GB) and 55 conversations
- 0.25: the MHA case (13) and why GQA shrinks the cache

### 4(b) Not recomputing the history

**Answer.** Any two of:
- **Prefix caching on the GPU** (vLLM automatic prefix caching or SGLang RadixAttention): keep the
  conversation's blocks after the turn ends and reuse them when the next turn arrives with the same
  prefix. Limit: HBM is small, and the blocks are evicted (LRU) long before the user returns hours later.
- **Hierarchical KV cache** (offload to CPU DRAM, SSD or a remote store, e.g. LMCache, Mooncake): much more
  capacity, so the history survives until the next turn. Limit: loading costs PCIe or network bandwidth,
  and it must be cheaper than recomputing. For an 8B model, the break-even bandwidth is about 4 GB/s, so
  even a local SSD is usually fast enough.
- **Cache-aware routing**: send the user back to the replica that already holds the history. Limit:
  replicas holding popular prefixes get overloaded, so routing must balance cache hits against load.

**Rubric.**
- 0.25 each for two correct techniques, 0.25 for a correct limit on at least one

---

## Set C: The ideas behind your A6 code

### 5(a) Left padding

**Answer.**
1. With left padding, every row's **last token is at the same slot** (slot 4), so at each decode step every
   row's new token goes to the same next slot (5, 6, ...): the cache grows by one column for the whole batch.
   With right padding, X's next token would go to slot 3 while Y's goes to slot 5. The rows would write to
   different slots, and X's new tokens would land in the middle of its padding.
2. **Position 0**, because GPT-2 was trained with every text starting at position 0, and X must look the same
   as when it runs alone. So X's real tokens at slots 2, 3, 4 get positions 0, 1, 2 (position = slot − pad).
   X's tokens must never attend to **slots 0 and 1**, the padding: they hold no real token. The pad mask hides
   them, and the causal mask hides the future slots.
3. GPT-2 adds a learned **position embedding** to each token. With slot numbers, X's text "starts" at
   position 2, so every one of X's tokens gets a different embedding than when X runs alone, and the
   attention results change, so the generated tokens change. Nothing crashes because positions 2–4 are
   valid. Y has no padding, so its slot numbers and positions are the same, and Y is not affected.

**Rubric.**
- 0.25: left padding lines up every row's newest token, so all rows write to the same slot
- 0.25: X starts at position 0, and the pad mask hides slots 0–1
- 0.25: the shifted position embeddings explain why X is wrong and Y is right

### 5(b) Blocks that are already promised

**Answer.**
1. free now = 20 − 4 − 2 = **14**. Still promised to R1: ⌈100/16⌉ − 4 = 7 − 4 = 3. Still promised to R2:
   ⌈64/16⌉ − 2 = 4 − 2 = 2. So **5** of the free blocks are already promised.
   A needs ⌈120/16⌉ = 8 ≤ 14 − 5 = 9, so **A is admitted**. Now all 8 of A's blocks are promised too, even
   though it holds none yet. B needs ⌈60/16⌉ = 4 > 14 − 5 − 8 = 1, so **B waits**.
2. Under the simpler rule, A (8 ≤ 14) and B (4 ≤ 14) are both admitted, but at full length the four requests
   need 7 + 4 + 8 + 4 = 23 blocks, more than the 20 in the pool. It does not fail at admission because blocks
   are handed out **as sequences grow**. The 14 "free" blocks include 5 that R1 and R2 will claim later, so
   "free right now" overstates the room. The pool runs dry only after the sequences have grown, many decode
   steps later.
   Safe admission **promises** that every admitted request can finish without being evicted, so it has no
   preemption plan: running out means the promise was broken, which is a bug, and the engine stops with an
   error. Optimistic admission overcommits on purpose: it expects to run out, so it **preempts** a request,
   frees its blocks, and recomputes it later with one prefill.

**Rubric.**
- 0.25: free 14, 5 still promised, A admitted (8 ≤ 9), B waits (4 > 1, counting A's 8)
- 0.25: blocks are handed out as sequences grow, so "free now" includes blocks already promised; the shortage
  shows up later
- 0.25: safe mode promises no preemption, so running out is a bug; optimistic mode expects it and preempts

**Common mistake.** Forgetting A's 8 blocks when B is checked: A holds none yet, but they are already promised.
