# Mock Quiz 6: Assignment 6 (Serving GPT-2)

> 15 minutes per set. Closed book, no devices. 3 points per set.
> Answer as you would in a design interview: two to four sentences per part, with the reasoning.
> Every number you need is on this sheet. Do **Set A** first and check the answers, then do **Set B**. **Set C** is about the ideas behind your A6 code.
> 1 GB = 10^9 bytes.

---

## Set A

**The setting.** The restaurant-booking site from Quiz 5 is live. Every listing page asks the model for one
sentence, on demand. You serve **GPT-2 large** (774M parameters, **36 layers, width 1280, 20 heads**,
1,024-token context) in **bf16** with your Assignment 6 engine, on one of the team's 24 GiB cards.
After the weights and the workspace, **20 GB is left for the KV cache**.

A listing request is a **70-token prompt** (instruction line + attribute list) and **at most 40 new tokens**.

**Decode step time on this card** (one token for every sequence in the batch)

| batch B | 1 | 16 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| step time (ms) | 14.0 | 14.3 | 15.1 | 17.0 | 24.5 | 41.0 |

### 1. Sizing the cache (1.5 points)

**(a)** How many bytes of KV cache does one token take? How many listing requests fit in the 20 GB at once
with a **contiguous** cache that reserves the full 1,024-token context per request, and how many with a
**paged** cache at block size 16? Show your working.

**(b)** Product sets the TBT (time between tokens) SLO at **30 ms**. What `max_batch` would you set, and
what tokens/s do you expect? Compare with the contiguous cache. Product then says, "Paging fits about 9×
more requests, so we should get about 9× the throughput." What is your response?

### 2. The engine in production (1.5 points)

**(a)** To make room for a second product, the listings get only **1 GB** of KV. You serve **600 listing
requests that arrive at once** (prompts 60–80 tokens, outputs 10–40 tokens, block size 16, max batch 128):

| engine | time (s) | tok/s | mean lat (s) | p90 lat (s) | mean batch | preempt |
|---|---|---|---|---|---|---|
| static | 7.3 | 2,050 | 4.1 | 6.6 | 31.0 | 0 |
| continuous (safe) | 5.4 | 2,780 | 2.9 | 4.9 | 45.2 | 0 |
| continuous (optimistic) | 4.5 | 3,350 | 2.4 | 4.2 | 62.5 | 38 |

Your manager says: "Optimistic preempted 38 requests. Each of those did its work twice, so it is wasting
GPU time. Turn it off." Do you agree? Use the numbers, and explain why a preemption costs less than it looks.

**(b)** A new hire rewrites `PagedKVCache.update`. Every test that decodes **one** sequence passes, and a
batch of prompts **with the same length** still matches the uncached model. But in a batch with prompts of
**different lengths**, the longest prompt's output is correct and the **shorter prompts produce wrong
tokens**. What is the most likely bug, and why does it only show up for the shorter prompts?
(Hint: think about what the block table holds for a short row.)

---

## Set B

**The setting.** Same model and card. The team now also runs the screenwriting **autocomplete** on it:
many users are decoding at once, and each one watches the text appear.

### 3. Sharing and scheduling (1.5 points)

**(a)** Every listing prompt is a **fixed 50-token instruction line** followed by a 20-token attribute list.
You turn on vLLM-style **automatic prefix caching** (16-token blocks, each block's hash includes its
parent's hash). How many prompt tokens per request can be reused, and what fraction of the prompt is that?
Then a teammate adds `Generated at 2026-11-03 14:02:11` to the **start** of every prompt, for logging, and
the cache hit rate drops to 0%. Why, and how would you fix it?

**(b)** A new feature summarizes a whole screenplay scene: **1,000-token prompts**. About **100 autocomplete
users** are decoding at any time. When a summary request arrives, the engine runs its whole prefill in one
step together with the 100 decodes, and every autocomplete user sees the text stutter. Step time against
the number of tokens in one step:

| tokens in the step | 64 | 256 | 512 | 768 | 1,024 |
|---|---|---|---|---|---|
| step time (ms) | 15 | 18 | 27 | 38 | 50 |

Which Lecture 15 technique fixes the freeze? With a **30 ms TBT SLO**, what token budget τ would you pick,
and what does the summary request pay for it?

### 4. Design from scratch (1.5 points)

**(a)** A customer wants to self-host **Llama-3.1-8B** (32 layers, **8 KV heads** × head dim 128, bf16) for
chat with **8K-token** conversations, on one **80 GB** GPU. The weights take 16 GB, and 4 GB is left for
activations. How many 8K conversations fit at once? Someone proposes switching to an older model with the
**same size and shape but full multi-head attention (32 KV heads)**. How many fit then?

**(b)** The customer's users come back to the same conversation many times a day. Each new turn re-sends the
whole history. Name two ways to avoid recomputing the history's prefill, and say what limits each one.

---

## Set C: The ideas behind your A6 code

**The setting.** No code needed. These questions are about the design choices inside your A6 cache and
scheduler.

### 5. Inside the cache and the scheduler (1.5 points)

**(a)** Your contiguous KV cache runs two prompts in one batch: X has 3 tokens and Y has 5. The cache pads X on
the **left** with 2 empty slots, so both rows are 5 slots long.
1. Why pad on the left and not on the right? (Hint: where does each row's next token go during decoding?)
2. What position should X's first real token get, and which slots must X's tokens never attend to?
3. A teammate gives every token its slot number as its position, so X's tokens get positions 2, 3, 4. Nothing
   crashes and Y's output is correct, but X's output differs from running X alone. Why?

**(b)** Your paged engine uses **safe admission**: it admits a request only if the pool can hold the request's
**full length** (prompt + max new tokens), in blocks of 16. Blocks are handed out as a sequence grows, not all
at once, and a newly admitted request holds no blocks until its prefill. The pool has **20 blocks**, and two
requests are running:

| request | full length (tokens) | blocks held now |
|---|---|---|
| R1 | 100 | 4 |
| R2 | 64 | 2 |

Two requests wait in this order: A (full length 120 tokens), then B (60 tokens).
1. Is A admitted? Then B? Show your working.
2. A teammate simplifies the rule to "admit if the request's blocks fit in the blocks that are free right
   now." Nothing goes wrong at admission, but later the engine runs out of blocks. Why later and not at
   admission? Why is running out a bug in safe mode, while optimistic mode handles it?
