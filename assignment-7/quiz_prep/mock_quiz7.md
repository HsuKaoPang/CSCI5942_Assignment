# Mock Quiz 7: Assignment 7 (RLAIF with GRPO) and Lecture 19 (Post-training)

> Same format as Quiz 6: **three questions, 1 point each**, 15 minutes, closed book, no devices, no AI.
> The setting is the assignment itself. Every number you need is on the sheet. 1 GB = 10^9 bytes.
> Each set covers different concepts. Sets A and B follow the assignment, and Sets C and D follow the
> lecture. Together they cover every concept in the A7 README and the L19 slides (the answers file has
> the list). Do one set, check it, then do the next.

**The setting (all sets).** In Assignment 7 you trained the GPT-2 medium model from Assignment 5 (full
fine-tune on E2E, linked as `out/sft`) further with **GRPO**. The reward was **gemini-2.5-flash** with a
rubric. Each step takes 8 MRs. For each MR the policy samples a **group of G = 8** descriptions at
temperature 1, until EOT or 64 new tokens. The judge scores all eight in **one call**, from 1 to 10. The
advantage is Â_i = (r_i − mean(r)) / (std(r) + 10⁻⁴). The per-token loss is −Â_i · log π_θ + β · k3, with
k3 = e^d − d − 1 and d = log π_ref − log π_θ, averaged over each completion's tokens and then over the
group. There is one optimizer step per batch. The measurements are from a reference run on one A40.

---

## Set A: The log-probs, the judge, and the update

### 1. The batch for the log-probs (1 point)

The prompt has P = 3 tokens `[p0 p1 p2]`. The group has two completions (G = 2 here to keep it small).
E is the EOT token, which is also used as padding.

- Completion A = `[a0 a1 a2 E]` (4 tokens, it stopped)
- Completion B = `[b0 E]` (2 tokens, it stopped)

(i) Write the two rows of token IDs that go into the model in one forward pass, and the mask over the
completion tokens.

(ii) The model returns one logit vector for each input position, 0 to 6. Which position's logits give
log π(a0)? Which give log π of A's final E? Which positions' logits are not used at all?

(iii) Row B has two padding tokens after its E. Do they change the log-probs of b0 and B's E? Why can
padding go on the **right** here, when in Assignment 6 the shorter prompts were padded on the left?

### 2. Measuring the judge (1 point)

Before training, `judge.py --part1` scored 30 test MRs on the `given` rubric. Each group had up to seven
human references plus one **intruder** (a good reference for a different MR). Every group was scored once
in order and once reversed.

| measure | value |
|---|---|
| human scores that moved by ≥ 2 when the order was reversed | 19 of 225 (8.4%) |
| groups where the intruder scored lowest | 30 of 30 |
| mean score of the human references | 7.75 |

(a) For each of the three numbers, say what it should be compared with and what it tells you.

(b) Late in training, a typical group scores **[9, 10, 9, 9, 10, 9, 9, 8]**. Using your answer to (a),
explain what the order effect does to the advantages in this group and to training. Give two ways to
reduce it, and what each one costs.

### 3. The update (1 point)

(a) Explain why the PPO ratio ρ = π_θ / π_old is exactly 1 in this loop, so the clip never fires. A
teammate wants to reuse each batch of samples for **four** optimizer steps. What changes, and what does
the clip do then? Why is reusing samples tempting? (The log says a step is 7.2 s, of which 5.5 s is
sampling and 0.9 s is judging.)

(b) GRPO training needs two engines: a fast generation engine (vLLM-style) to sample the groups, and the
training engine to update the weights. Discuss two ways to place them on the GPUs, and the trade-off
between them.

---

## Set B: Advantages, evaluation, and the rubric

### 1. Advantages by hand (1 point)

Use G = 4 and std = the population standard deviation.

(i) A group scores **[9, 7, 9, 7]**. What are the mean, the std and the four advantages? Which
completions does the update make more likely?

(ii) A kinder judge would have scored the same group **[10, 8, 10, 8]**, and a stricter one
**[10, 6, 10, 6]**. What are the advantages now? What does this tell you about which judge errors matter
for GRPO and which do not?

(iii) A group scores **[8, 8, 8, 8]**. What are the advantages, and what is left of the loss? A new hire
samples with **argmax** instead of `torch.multinomial`. What happens to every group, and to training, at
step 0?

### 2. Reading the evaluation (1 point)

`eval.py` judges each policy's greedy description together with seven human references (rank 1 is the
best of 8). The last four columns are metrics **the judge never saw**. The run **long** is a teammate's
extra run with β = 0 for 400 steps.

| run | judge (95% CI) | humans' mean | rank of 8 | BLEU | slot coverage | mean tokens | FineWeb loss |
|---|---|---|---|---|---|---|---|
| sft | 8.75 ± 0.16 | 7.57 | 1.79 | 63.2 | 97.2% | 33.0 | 3.83 |
| judge-leash (β = 0.04, 100 steps) | 9.42 ± 0.08 | 7.55 | 1.06 | 60.1 | 99.8% | 32.5 | 3.85 |
| long (β = 0, 400 steps) | 9.81 ± 0.04 | 7.50 | 1.00 | 38.0 | 100% | 61.0 | 4.70 |

A greedy description from **long**, which the judge scored 9:
> *The Vaults is a pub near Café Adriatic with a price range of more than £30 and a customer rating of 5
> out of 5, making The Vaults a pub near Café Adriatic with a customer rating of 5 out of 5.*

(a) For each of the four unseen columns, say what it measures and what it is compared with. What would
the rank of 8 be for a description that is no better than a human reference?

(b) Product says: "**long** has the best judge score, rank 1.00 and 100% slot coverage. Ship it." Using
your answer to (a), respond, and name what happened. Which run would you ship?

### 3. The rubric is the reward (1 point)

(a) To stop the repetition, a teammate rewrites the rubric as: *"Please judge carefully how well this
text describes all the attributes of the restaurant, as a good restaurant guide would."* Why will this
probably not stop the repetition? What should the new rubric do instead, and how would you check after
training that it worked?

(b) Product asks: "Why use an LLM judge at all? A program could check that every slot value appears in
the text." Discuss the two kinds of reward, the LLM judge and the program checker, and the trade-off.
Include what each one rewards that it should not, and how much faster a step would get with the checker.

---

## Set C: Post-training methods

Byte convention from lecture: a **trained** model needs bf16 weights + bf16 gradients + fp32 Adam m and v
= **12 B per weight**. A **frozen** model needs bf16 weights only = **2 B per weight**. One A40 has
**46 GB**. σ(0) = 0.50, σ(1) = 0.73, σ(2) = 0.88, σ(3) = 0.95.

### 1. Four models, two, or one (1 point)

The team considers a **3B** policy.

(i) PPO-based RLHF keeps a policy, a value model, a reference model and a reward model, all 3B. Which of
them are trained, and how many GB do the four need?

(ii) GRPO with your API judge: which models are left, and how many GB? Does it fit on one A40? What is
still missing from your count?

(iii) What did the value model do in PPO, what replaces it in GRPO, and what does GRPO pay instead?

### 2. Preferences without a reward model (1 point)

(a) A reward model is trained on 20,000 pairs of listings that users compared. For one MR it scores
listing A at 3.1 and listing B at 1.1. Under Bradley–Terry, what is P(A preferred over B)? A teammate adds
**+5** to every reward of that MR. What changes in P(A preferred over B), and what changes in a GRPO
update?

(b) Using your answer to (a), explain why **DPO** can train on the same 20,000 pairs without a reward model
and without sampling. What does DPO give up compared with GRPO?

### 3. Why RL after SFT, and which method (1 point)

(a) Why can SFT alone not make the model write better than its references? In InstructGPT, name the three
steps, and explain why outputs of the 1.3B RLHF model were preferred over outputs of the 175B SFT model.

(b) The team has **5,000 thumbs-up / thumbs-down** labels on single listings, with no pairs. DPO needs
pairs. Discuss two ways to use this data, and the trade-off between them.

---

## Set D: The leash, verifiers, and recipes

### 1. The KL estimate by hand (1 point)

Use e^0.69 = 2, e^−0.69 = 0.5 and ln 2 = 0.69. d = log π_ref − log π_θ, and k3 = e^d − d − 1.

(i) Compute k3 for one token in three cases: (1) π_θ = 0.5 and π_ref = 0.5; (2) π_θ = 0.25 and
π_ref = 0.5; (3) π_θ = 0.5 and π_ref = 0.25.

(ii) Can k3 be negative? Why is k3 used instead of the plain log ratio log π_θ − log π_ref?

(iii) In your runs, `judge-leash` (β = 0.04) and `judge-noleash` (β = 0) ended almost the same: mean KL
0.15 and 0.16 at step 99, judge scores 9.42 and 9.43. Using β and these KL values, explain why the leash
did almost nothing in 100 steps. When would it start to matter?

### 2. The proxy and the people (1 point)

An engineer runs a long GRPO run against a **learned reward model** and has people rate 200 descriptions
(1–5) every few hundred steps.

| KL to SFT (nats) | 0 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| reward model score | 6.7 | 8.0 | 8.8 | 9.4 | 9.8 |
| people's rating (1–5) | 3.1 | 3.6 | 3.8 | 3.7 | 3.2 |

(a) Where would you stop, and why does the reward model keep rising while people's ratings fall?

(b) Using your answer to (a), what does β do about it? What did DeepSeekMath do to the reward model when
RL against it stopped improving?

### 3. Verifiers and the R1 recipe (1 point)

(a) A math helper is trained with GRPO and one of two checkers. **Naive** gives reward 1 if the gold
number appears anywhere in the answer. **Strict** gives reward 1 only if the last line is exactly
`#### <gold>`.

| run | mean training reward (its own checker) | held-out accuracy (strict checker) |
|---|---|---|
| base | — | 15% |
| trained with naive | 0.75 | 19% |
| trained with strict | 0.58 | 63% |

The engineer says: "Naive gets the higher reward, so it trains the better model." What is wrong with
this comparison, and what did the naive policy most likely learn? How does this connect to the judge in
Assignment 7?

(b) DeepSeek-R1-Zero was trained with RL from the base model with two rule rewards. Name them. Why did
DeepSeek-R1 add a cold-start SFT stage first? For a small model, discuss the two options, RL it directly
or distill from R1, and the trade-off.
