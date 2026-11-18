# Mock Quiz 7: Answers and Rubrics

The rubrics copy the style of the real Quiz 6 grading: each item is one idea with its points, and the
items add up to 1 point per question. Quiz 6 rubrics rewarded the **mechanism**, the **payoff**, and
**each option plus the trade-off**. Every part ends with 中文重點 (key points in Chinese).

**What Quiz 6 taught us (your 1.2/3):**
- **Q1 traced the core data structure by hand.** You lost points by assuming a new block every step. Here
  the matching trap is "which logit predicts which token", so practise Set A Q1 and Set B Q1 until you can
  do them without thinking.
- **"Using your answer to (a)"** means the grader wants (a)'s number inside (b)'s answer.
- **The last part is a lecture idea framed as "two options + trade-off".** Name both options, say what
  each one does, then say what you trade.

---

## Set A

### A1 (i) The two rows and the mask (0.3)

T = the longest completion = 4, so each row has P + T = 7 positions.

```
position:   0   1   2   3   4   5   6
row A:     p0  p1  p2  a0  a1  a2  E
row B:     p0  p1  p2  b0  E   E   E      <- the last two E are padding
mask A:                 1   1   1   1
mask B:                 1   1   0   0
```

B's first E is **real**: the policy chose to stop there, so it counts in the loss (mask = 1). The two E
after it are padding (mask = 0).

**Rubric:** 0.15 for rows of length P + T with E padding on the right · 0.15 for the mask, including that
B's own E counts and its padding does not.

**中文重點：** 每一列長度是 P + T = 3 + 4 = 7，短的那篇在**右邊**補 E。mask 只蓋在 completion 上：B 自己產生的 E 是真的（它決定停下來，要算進 loss），後面補上去的 E 才是 0。

### A1 (ii) Which logits predict which token (0.3)

The logits at position t predict the token at position t + 1.
- log π(a0): a0 sits at position 3 (= P), so it is read from the **logits at position 2 (= P − 1)**.
- A's final E sits at position 6, so it is read from the **logits at position 5**.
- Logits used: positions 2, 3, 4 and 5 (P − 1 to P + T − 2), which gives T = 4 rows. **Not used:** positions
  0 and 1 (they predict prompt tokens, which are not the policy's choice) and position 6 (it would predict
  the token after the end).

**Rubric:** 0.1 for position 2 for a0 · 0.1 for position 5 for the final E · 0.1 for 0, 1 and 6 unused, with
the reason.

**中文重點：** 位置 t 的 logits 預測的是位置 t+1 的 token，所以要往左移一格。completion 第一個 token 在位置 P，它的機率是從位置 **P−1** 的 logits 讀出來的。prompt 本身的 token 不是 policy 選的，所以不算；最後一個位置的 logits 預測的是「結束之後」的 token，用不到。（這題就是 Quiz 6 的 block table：一步一步推出 index。）

### A1 (iii) Why right padding is harmless here (0.4)

The padding does **not** change b0's or B's E's log-probs. The causal mask means a position only attends
to itself and earlier positions. b0 (position 3) and B's E (position 4) are read from the logits at
positions 2 and 3, which never see positions 5 and 6. The padding's own log-probs are computed but masked
out of the loss. In Assignment 6 the prompts had **different lengths**, and decoding appends each new
token at the end of every row, so padding went on the left to keep every row's last token in the same
column (with positions and masks fixed up). Here every row shares **the same prompt**, so the real tokens
already line up from position 0. Padding goes after the content, where the causal mask hides it.

**Rubric:** 0.2 for "no change: the causal mask means earlier positions never attend to later ones; the
padding is masked out of the loss" · 0.2 for "same prompt, so the rows already line up; padding after the
content is hidden by causality (A6 padded on the left because prompts differed and new tokens are appended)".

**中文重點：** causal mask 讓每個位置只看得到自己和前面，所以接在後面的 padding 影響不到前面的 token；padding 自己的機率會算出來，但被 mask 掉。A6 用左邊補齊，是因為每個 prompt 長度不同，decode 時新 token 又要接在最後面。這裡同一組共用同一個 prompt，本來就對齊了，補在右邊就好。

### A2 (a) What to compare each number with (0.4)

- **8.4% moved by ≥ 2 on reversal → the size of the judge's noise.** Compare it with the gaps the policy
  needs to see between candidates. A perfect judge would be 0%.
- **Intruder lowest 30/30 → the judge reads the MR.** Compare it with **chance**: with about 8 candidates, a
  judge that ignores the MR puts the intruder lowest about 1/8 of the time, which is about 4 of 30.
- **Humans' mean 7.75 → the ceiling and the headroom.** The references are human writing, so 7.75 means
  the policy has room to gain. If the policy later scores far above the humans, ask whether it writes
  better or has found what the rubric rewards.

**Rubric:** 0.15 for intruder vs chance (about 1/8, 4 of 30) · 0.15 for the humans' mean as the ceiling
and headroom (beating it is suspicious) · 0.1 for 8.4% as noise with 0% as ideal.

**中文重點：** 每個數字都要說「跟什麼比」：入侵者 30/30 vs 亂猜大約 4/30（1/8）→ judge 真的有讀 MR；人類平均 7.75 → 天花板，還有進步空間，但如果 policy 超過人類很多，就要懷疑是在鑽 rubric 的漏洞；8.4% → judge 的雜訊有多大（理想是 0%）。

### A2 (b) What the order effect does to the advantages (0.6)

The group [9, 10, 9, 9, 10, 9, 9, 8] has mean 9.125 and a spread of max − min = 2. That is **the same size**
as the move that 8.4% of scores make when only the order changes. A candidate's score, and so the sign of
its advantage, can depend on **where it sits in the list** and not on how good it is. Then the update
pushes up some completions only because of their position. The gradient is noisier, and learning slows
down late in training, when the real differences are small.

Two fixes:
1. **Score each group twice, once in each order, and average** (or shuffle the order every call). This
   costs 2× the judge calls.
2. **Score pointwise:** one call per candidate, so there is no list position at all. This costs 8× the
   calls, and you lose the side-by-side comparison, so the scores become less consistent across the group.

(A larger judge is another option, at a higher cost per call.)

**Rubric:** 0.2 for "the spread (about 2) equals the order effect, using (a)" · 0.2 for "the advantage's
sign can come from position → noisy gradient, slower learning" · 0.2 for two fixes, each with its cost.

**中文重點：** 一定要用 (a) 的數字：這組的 max − min = 2，剛好等於「換個順序就會動 2 分」的雜訊，所以 advantage 的正負號可能只是位置造成的，學到錯的東西，訓練變慢。修法：(1) 正反兩個順序都評一次再平均（呼叫次數 ×2）；(2) 一篇一篇分開評（×8，而且少了互相比較）。

### A3 (a) Why ρ = 1, and what four updates change (0.5)

ρ = π_θ / π_old compares the policy being updated with the policy that **sampled** the batch. In our loop
we sample with the current weights and then take **one** optimizer step, so at the moment the loss is
computed, π_θ = π_old and ρ = 1. The clip (keep ρ within 1 ± ε) never activates. With **four** steps per
batch, after the first step θ has moved, so the samples now come from an **older** policy and ρ ≠ 1. The
loss multiplies the advantage by ρ, and the clip **stops the gradient** for a token once its ratio has moved
past 1 ± ε in the direction the advantage pushes. This keeps the policy from moving too far on stale
samples. It is tempting because **generation dominates** a step: 5.5 s of 7.2 s is sampling, so reusing a
batch four times cuts the sampling cost per update by about 4×.

**Rubric:** 0.2 for "sampled from the same weights, one step → π_θ = π_old" · 0.15 for "after step 1 the
samples are stale, so the clip stops a token's gradient once ρ leaves 1 ± ε" · 0.15 for "generation
dominates (5.5 of 7.2 s), so reuse saves time".

**中文重點：** ρ 比的是「現在要更新的 policy」和「當初抽樣的 policy」。我們抽完樣只更新一步，兩個是同一個，所以 ρ = 1，clip 永遠不會觸發。如果同一批資料更新 4 次，第 2 步開始資料就變「舊」了，ρ ≠ 1，clip 會在某個 token 的機率變太多時停止推它。想重複使用，是因為抽樣佔了 5.5/7.2 秒。

### A3 (b) Two ways to place the two engines (0.5)

1. **Colocated:** the same GPUs are time-shared. The generation engine samples, then the trainer updates,
   then the new weights are loaded into the generation engine in GPU memory. No GPU sits idle waiting for
   another pool, and weight transfer is cheap. But both engines must fit in the same memory (the KV cache
   competes with optimizer state), and the phases run **one after the other**.
2. **Disaggregated:** separate GPU pools for generation and training, and the new weights are sent every
   step over the network (NCCL broadcast). Each pool can be sized and tuned for its own job, and generation
   can overlap with training (asynchronous rollouts, as in OpenRLHF or AReaL). But you pay for the weight
   transfer, one pool can idle while the other works, and with asynchronous rollouts the samples come from
   an **older** policy, so you are off-policy and need the ratio and clip from (a).

**Trade-off:** colocated uses memory and GPUs efficiently but runs the phases serially. Disaggregated runs
in parallel and scales, but costs weight transfer, possible idle time, and stale samples. (On one GPU, as in
A7, there is no choice, and the wait is generation.)

**Rubric:** 0.15 for colocated (same GPUs, time-shared, weights reloaded) · 0.15 for disaggregated (separate
pools, weights broadcast) · 0.2 for the trade-off (memory sharing and serial phases vs transfer, idle time
and off-policy samples).

**中文重點：** 兩種擺法：**colocated**（同一批 GPU 輪流做生成和訓練，權重在 GPU 內換，省資源但要分記憶體、只能輪流跑）vs **disaggregated**（生成和訓練各用一批 GPU，每步用 NCCL 把新權重廣播過去，可以平行、非同步，但要付傳輸成本，而且非同步時樣本是舊 policy 產生的 → off-policy，需要 ratio + clip）。

---

## Set B

### B1 (i) The advantages for [9, 7, 9, 7] (0.3)

Mean = 8. Every deviation is ±1, so std = 1. Advantages = (±1) / (1 + 10⁻⁴) ≈ **[+1, −1, +1, −1]**. The
update makes completions 1 and 3 (the 9s) **more likely** and completions 2 and 4 **less likely**.

**Rubric:** 0.15 for mean 8 and std 1 · 0.15 for advantages ±1 and which ones go up.

**中文重點：** 平均 8，每篇都差 ±1，標準差 1，advantage ≈ [+1, −1, +1, −1]。拿 9 分的機率被推高，拿 7 分的被壓低。

### B1 (ii) A kinder judge and a stricter judge (0.3)

[10, 8, 10, 8]: mean 9, std 1 → **[+1, −1, +1, −1]**, the same. [10, 6, 10, 6]: mean 8, std 2 → (±2)/2 =
**[+1, −1, +1, −1]**, the same again. Subtracting the mean removes any constant shift, and dividing by the
std removes any scale. So a judge that is **consistently** too kind or too strict does not matter. What
matters is the **ranking and the relative gaps within one group**, so errors that change the order inside
a group (position bias, noise) are the ones that hurt.

**Rubric:** 0.15 for both give the same advantages · 0.15 for "shift and scale cancel; only order inside a
group matters, so position bias and noise hurt".

**中文重點：** 減平均 → 全部加減一個常數沒差；除以標準差 → 全部放大縮小也沒差。所以 judge 整體偏寬或偏嚴都不影響，**會影響的是同一組裡面的排名**被打亂（位置偏差、雜訊）。

### B1 (iii) A flat group, and argmax (0.4)

[8, 8, 8, 8]: std = 0, so every advantage = 0 / 10⁻⁴ = **0**. The −Â · log π term is zero, so the only thing
left is **β · k3**, which just pulls the policy back toward the reference. With **argmax**, all G samples
of a prompt are **identical** (the same greedy text), so the judge gives them the same score, and **every
group is flat**. At step 0 the policy equals the reference, so k3 = 0 too: the loss and the gradient are
exactly zero, and nothing ever changes (the log would show group std 0, every group flat, a constant
reward, and mean KL stuck at its step-0 value). GRPO needs **different** samples to compare, so it must
sample with temperature through `torch.multinomial`.

**Rubric:** 0.1 for advantages 0 · 0.1 for "only β · k3 is left, toward the reference" · 0.2 for "argmax →
identical samples → every group flat → no learning; at step 0, k3 = 0 too".

**中文重點：** 全部同分 → advantage 全是 0 → 只剩 β·k3（把 policy 拉回 reference）。用 argmax 的話，同一題的 G 篇一模一樣，分數一定一樣，每組都是平的；而且第 0 步 policy = reference，k3 也是 0 → 梯度完全是 0，永遠學不到東西。GRPO 一定要有隨機抽樣，同一組裡才會有差異可以比較。

### B2 (a) What each unseen column measures (0.4)

- **BLEU:** n-gram overlap with the **human references**. Compare it with sft (63.2): a drop means the
  wording drifts away from how people wrote it, which is not necessarily worse.
- **Slot coverage:** the share of MR attributes that appear in the text. It checks the facts, and 100% is
  the maximum.
- **Mean tokens:** length. Compare it with sft (33) and the humans. A big jump means padding or repetition.
- **FineWeb loss:** loss on general web text, which measures **forgetting**. Compare it with sft (3.83):
  higher means the model is losing general language.

**Rank of 8:** a description exactly as good as a typical human reference would land in the middle,
**about 4.5** (the average of 1 to 8). 1.0 means the judge put it first in every group.

**Rubric:** 0.25 for what the four columns measure, each with its reference point (sft or the maximum) ·
0.15 for rank 4.5 as the "no better than a human" point.

**中文重點：** BLEU = 跟人類參考答案的字詞重疊；slots = 屬性有沒有都寫到；tokens = 長度；FineWeb loss = 有沒有忘掉一般語言（遺忘）。全部都跟 sft 那一列比。rank of 8 的參考點是 4.5（跟人類一樣好就會排在中間）。

### B2 (b) "Ship long" (0.6)

No. Using the reference points from (a): the judge column improved, but **every unseen column got worse**.
BLEU fell from 63 to 38, mean tokens almost doubled (33 → 61), and FineWeb loss rose from 3.83 to 4.70, so
the model is **forgetting** general language. The sample shows why: it states the slots and then
**repeats them**. That is fully covered (100% slots) and still scored 9, but no restaurant guide would print
it. The humans' mean stayed at 7.5 while the policy got 9.8, so the policy "beat" people by finding what
this judge rewards. This is **reward hacking / overoptimization** of a proxy reward (Gao's curve). With
β = 0 and 400 steps, nothing held it near the SFT model. **Ship judge-leash:** a better judge score with
small changes in BLEU, length and FineWeb.

**Rubric:** 0.2 for three or more unseen columns worse, with numbers (BLEU, tokens, FineWeb = forgetting) ·
0.2 for "reward hacking: the sample repeats, and the proxy is overoptimized" · 0.2 for the verdict (ship
leash) and why (β = 0 for 400 steps, so no leash).

**中文重點：** 用 (a) 的參考點：judge 變高，但**所有 judge 沒看到的指標都變差**（BLEU 63→38、長度變兩倍、FineWeb 3.83→4.70 = 遺忘）。範例在重複同一句話，slots 100% 但沒有人會這樣寫。這就是 **reward hacking / 過度最佳化**（Gao 曲線）：β = 0 又跑 400 步，沒有牽繩拉住。要上線的是 leash。

### B3 (a) Why the paraphrase fails (0.5)

The rubric **is** the reward. The paraphrase asks the same thing in different words ("describes all the
attributes … as a good guide would"), so the judge will reward the same behaviour, and the repetition
still states every attribute. To change the reward, the rubric must **name the behaviour you do not
want**, for example "each attribute appears once; no repeated phrases; at most N words". To check that it
worked: (1) rerun the **Part 1 test** on the new rubric (intruder vs chance, order noise, humans' mean), (2)
train and read **samples.jsonl** for the old hack and for new hacks, and (3) evaluate on the **old** rubric
and on the **unseen metrics** (tokens, BLEU, FineWeb), not only on the new rubric, which the policy was
trained to please.

**Rubric:** 0.2 for "the rubric is the reward; a paraphrase is the same reward" · 0.15 for "name the
unwanted behaviour explicitly" · 0.15 for a check after training (samples, the Part 1 rerun, unseen metrics
or the old rubric).

**中文重點：** rubric 就是 reward。換句話說還是同一個 reward，重複的句子照樣拿高分。要改就要**明寫你不要的行為**（每個屬性只出現一次、不能重複）。驗證方法：新 rubric 先跑 Part 1、訓練完看 samples.jsonl 有沒有舊的或新的 hack、用舊 rubric 和 judge 沒看到的指標評（不能只用新 rubric 評自己）。

### B3 (b) Judge vs checker (0.5)

- **LLM judge:** it can reward fluency and naturalness, things a program cannot check. But it is noisy and
  position-biased (8.4%), costs money per call (800 calls per run), and can be hacked, as with the
  repetition and the venue types made up for MRs without eatType.
- **Program checker** (a verifiable reward): it is deterministic, free and instant, with no position bias.
  But it only checks what it can test, here that a slot string appears. It would reward **listing or
  stuffing the slot values**, repeating them, or awkward template text, as long as the strings are present.
  It cannot see hallucinated extra facts unless it checks for them too.
- **Speed:** judging is 0.9 s of a 7.2 s step, so removing it saves **at most about 12%**. Generation
  (5.5 s) dominates, so the bigger win is a faster generation engine (vLLM: paged KV, continuous batching).

**Trade-off:** the checker is cheap and hard to fool on what it checks, but blind to quality. The judge
sees quality but is costly, noisy and hackable in other ways.

**Rubric:** 0.15 for the judge's strengths and weaknesses · 0.15 for the checker's, including what it would
wrongly reward · 0.2 for the speed-up (at most about 12%, generation dominates) and the trade-off.

**中文重點：** judge 能評「自然、通順」，但有雜訊、有位置偏差、要錢，也會被鑽漏洞。checker 準確、免費、沒有偏差，但只看得到它檢查的東西：只要字串都出現就給分 → 會鼓勵條列、重複、硬塞字串。速度：judge 只佔 0.9/7.2 ≈ 12%，**生成才是瓶頸**。

---

## Set C

### C1 (i) PPO's four models (0.4)

The **policy** and the **value model** are trained: 3B × 12 B = 36 GB each, so 72 GB. The **reference** and
the **reward model** are frozen: 3B × 2 B = 6 GB each, so 12 GB. Total **84 GB**.

**Rubric:** 0.2 for which two are trained · 0.2 for 84 GB with the working shown.

**中文重點：** 要訓練的是 policy 和 value model（各 36 GB），凍結的是 reference 和 reward model（各 6 GB），總共 84 GB。

### C1 (ii) GRPO with an API judge (0.3)

Left: the policy (36 GB, trained) and the reference (6 GB, frozen) = **42 GB**. The value model is gone, and
the reward model is replaced by the API. 42 ≤ 46, so the **weights and optimizer state fit**, but only about
4 GB is left. **Missing:** activations for the backward pass, and the KV cache for sampling G completions
per prompt (plus the generation engine's own copy of the weights if it is separate). So it is tight and
probably needs a smaller batch, activation checkpointing, or sharding.

**Rubric:** 0.15 for 42 GB, fits on paper · 0.15 for what is missing (activations, KV cache for sampling).

**中文重點：** 剩 policy（36）+ reference（6）= 42 GB，帳面上放得進 46 GB，但只剩 4 GB，還沒算 activation 和抽樣時的 KV cache，所以很緊。

### C1 (iii) What replaced the value model (0.3)

In PPO the value model predicts **the expected reward from each token onward**. It is the **baseline**: the
advantage is the reward minus what was expected, computed per token (GAE). GRPO replaces it with **the
mean reward of the G samples for the same prompt**. It pays with **more samples per prompt** (G
generations, and generation dominates the step), and with **one advantage shared by every token** of a
completion, with no per-token credit assignment.

**Rubric:** 0.1 for the value model as a per-token baseline · 0.1 for the group mean as the replacement ·
0.1 for the cost (G samples per prompt, no per-token credit).

**中文重點：** value model 是基準線（預期能拿幾分）。GRPO 改用同一題 G 篇的平均分數當基準線。代價：每題要生成 G 篇（生成本來就最慢），而且整篇的每個 token 共用同一個 advantage。

### C2 (a) Bradley–Terry and +5 (0.4)

P(A ≻ B) = σ(r_A − r_B) = σ(3.1 − 1.1) = σ(2) = **0.88**. Adding +5 to every reward gives
σ(8.1 − 6.1) = σ(2) = 0.88, **unchanged**, because Bradley–Terry only uses the **difference**. In GRPO the
+5 is also **removed by subtracting the group mean**, so the advantages and the update do not change.

**Rubric:** 0.15 for 0.88 · 0.1 for unchanged, only differences matter · 0.15 for the GRPO update unchanged
(the mean subtraction).

**中文重點：** σ(3.1 − 1.1) = σ(2) = 0.88。全部 +5 → 差距不變 → 機率不變；GRPO 減掉組內平均，+5 也被消掉，更新不變。

### C2 (b) Why DPO needs no reward model and no sampling (0.6)

Using (a): Bradley–Terry only needs the **difference** of two rewards. The KL-regularized RL objective has a
closed-form best policy, which lets the reward be written in terms of the policy itself:
r(x, y) = β log(π_θ(y|x) / π_ref(y|x)) + β log Z(x). This is the **implicit reward**. In the difference for
two answers to the **same** prompt, β log Z(x) **cancels**. The loss becomes −log σ(β [log-ratio of the
chosen answer − log-ratio of the rejected one]), which only needs the policy's and the reference's
log-probs **on the stored pairs**. So there is no separate reward model, and no sampling: it is supervised
training on fixed data. **What it gives up:** it is **offline**. It never sees or corrects its **own**
outputs, so it cannot explore beyond the dataset, and it can overfit the pairs. GRPO samples on-policy and
can improve on what it actually writes, at the cost of generation and a reward source.

**Rubric:** 0.2 for the implicit reward β log π/π_ref · 0.15 for "Z(x) cancels in the difference (from
(a))" · 0.1 for "only log-probs on stored pairs → no sampling" · 0.15 for the trade-off (offline, no
exploration, vs GRPO's on-policy samples).

**中文重點：** 接 (a)：BT 只看差距。DPO 把 reward 寫成 β·log(π/π_ref) + β·log Z(x)，同一題兩個答案相減時 Z(x) 消掉，所以只要算存好的資料上的機率，不用 reward model，也不用抽樣。代價是 **offline**：從來不看自己寫出來的東西，沒有探索；GRPO 是 on-policy，可以針對自己的輸出改進。

### C3 (a) The SFT ceiling and InstructGPT (0.5)

SFT minimizes cross-entropy to the references: it rewards **matching** what people wrote. There is **no
term for "better than the reference"**, so the best SFT can do is imitate the data, mistakes included.
InstructGPT's three steps: (1) **SFT** on demonstrations, (2) train a **reward model** on human comparisons
(Bradley–Terry), (3) **RL (PPO)** against the reward model with a KL penalty to the SFT model. People
preferred outputs from the **1.3B** RLHF model over the **175B SFT** model (win rate above 0.5) because
comparing is easier than writing. Labelers can say which of two outputs is better even when they could not write it. RL then
optimizes **what people prefer** directly, which is worth more than 100× the parameters trained only to
imitate.

**Rubric:** 0.2 for "SFT imitates; no term for better" · 0.15 for the three steps · 0.15 for "comparisons
are easier than demonstrations; RL optimizes preference directly → 1.3B beats 175B".

**中文重點：** SFT 只學「跟參考答案一樣」，沒有「比參考答案更好」這一項 → 天花板就是資料本身。InstructGPT 三步：SFT → 用比較資料訓練 reward model → PPO + KL。1.3B 贏 175B，因為人「比較」比「寫」容易，RL 直接最佳化人的偏好。

### C3 (b) 5,000 thumbs labels (0.5)

1. **KTO:** it learns directly from **unpaired** good/bad labels, with a reference model. Each label pushes
   its listing's implicit reward up or down relative to a baseline. It is offline and simple, uses the data
   as it is, and needs no reward model.
2. **Train a reward model** (a classifier: P(thumbs up | listing)) and use it as the **reward for GRPO**.
   It is online, so the policy explores and learns from its own outputs, but it costs generation and a
   second model, and the policy can **hack the learned reward** (overoptimization), so it needs the KL
   leash and checks.

(A third option is to pair up liked and disliked listings of the same MR and run DPO, but only where both
exist.)

**Trade-off:** KTO is cheap and stable but offline, limited to what is in the data. The reward model plus
RL can go further but is expensive and hackable.

**Rubric:** 0.15 for KTO on unpaired labels · 0.15 for reward model + RL · 0.2 for the trade-off (offline
and cheap vs online, exploring and hackable).

**中文重點：** 只有讚或倒讚、沒有成對 → (1) **KTO**：直接吃單篇好或壞的標籤，offline、便宜；(2) 先訓練一個 reward model（分類器），再用 GRPO：會探索，但貴，而且 reward model 會被鑽漏洞，要用 KL 拉住。（DPO 家族還有：IPO 防過擬合、ORPO 和 SimPO 不需要 reference model。）

---

## Set D

### D1 (i) k3 by hand (0.3)

1. d = ln 0.5 − ln 0.5 = 0 → k3 = e⁰ − 0 − 1 = **0**.
2. d = ln 0.5 − ln 0.25 = ln 2 = 0.69 → k3 = 2 − 0.69 − 1 = **0.31**.
3. d = ln 0.25 − ln 0.5 = −0.69 → k3 = 0.5 + 0.69 − 1 = **0.19**.

**Rubric:** 0.1 for each case.

**中文重點：** d = ln π_ref − ln π_θ。一樣 → 0；policy 比 ref 低一半 → 2 − 0.69 − 1 = 0.31；policy 比 ref 高一倍 → 0.5 + 0.69 − 1 = 0.19。

### D1 (ii) Why k3 (0.3)

k3 is **never negative**: e^d ≥ 1 + d for every d, with equality only at d = 0. The plain log ratio
log π_θ − log π_ref (k1) is an unbiased per-token estimate of the KL too, but on any one sample it can be
**negative** and has **high variance**. As a penalty, a negative value would reward drifting. k3 is
unbiased (because E over π_θ of e^d = 1) and has lower variance, so every token's penalty points the same
way: back toward the reference.

**Rubric:** 0.15 for "never negative (e^d ≥ 1 + d)" · 0.15 for "k1 can be negative per sample and is noisy;
k3 is unbiased with lower variance".

**中文重點：** e^d ≥ 1 + d，所以 k3 永遠 ≥ 0。直接用 log 比值（k1）單一個樣本可能是負的，變成「鼓勵跑遠」，而且很吵。k3 不偏又比較穩。

### D1 (iii) Why the leash did nothing in 100 steps (0.4)

The KL term in the loss is β · KL = 0.04 × 0.15 ≈ **0.006** per token. The advantage term is of order 1
(the advantages are standardized to std 1). So the leash's pull is about **100× weaker** than the reward's
push, and with only 100 steps at a learning rate of 5 × 10⁻⁶, the policy never moved far enough (KL about
0.15) for the leash to bite. That is why leash and noleash end almost the same. It starts to matter on
**longer runs or with a higher learning rate**, when KL grows into the range where reward hacking appears
(the **long** run in Set B). There β · KL becomes comparable to the advantage term, and the leash stops the
policy at a point on Gao's curve before quality falls.

**Rubric:** 0.2 for β · KL ≈ 0.006 vs the advantage's order 1 · 0.1 for "the policy has not moved far (100
steps)" · 0.1 for "matters on long runs, when KL grows and hacking starts".

**中文重點：** 0.04 × 0.15 ≈ 0.006，advantage 那一項大約是 1，KL 的拉力小了大約 100 倍；100 步、lr 很小，policy 根本沒跑遠，所以有沒有牽繩都一樣。跑很久（像 Set B 的 long）KL 變大、開始鑽漏洞的時候，牽繩才會起作用。

### D2 (a) Where to stop (0.4)

Stop around **KL ≈ 10**, where people's rating peaks (3.8). After that the reward model keeps rising (8.8 →
9.8) while people fall (3.8 → 3.2). The reward model is a **proxy** trained on samples near the SFT
model. As the policy moves away, it reaches outputs the reward model never saw, where its errors are large,
and RL **exploits those errors**. This is **overoptimization** (Goodhart, Gao's curve): the proxy goes up
while the true quality goes down.

**Rubric:** 0.15 for stopping at about KL 10 · 0.25 for "proxy trained near SFT; off-distribution errors are
exploited → overoptimization".

**中文重點：** 在 KL ≈ 10 停（人的評分最高 3.8）。之後 reward model 繼續升、人反而降：reward model 只看過 SFT 附近的輸出，policy 跑遠以後，reward model 的錯誤就被利用了 → 過度最佳化（Gao 曲線）。

### D2 (b) β and DeepSeekMath (0.6)

Using (a): the objective is E[r] − β · KL(π_θ ‖ π_ref). β charges the policy for **moving away from the SFT
model**. A larger β makes the optimum sit at a **smaller KL**, so it settles near the peak at KL ≈ 10
instead of running to 40. A smaller β (or β = 0) lets it slide down the right side of the curve. But β does
not fix the reward model: it only limits how far you exploit it, so you still **monitor a gold signal**
(people, held-out metrics) and stop early. **DeepSeekMath** ran **iterative RL**: when a run stopped
improving, they **retrained the reward model on samples from the current policy** (keeping some old data as
replay), then continued GRPO from the new policy. The refreshed reward model knows the outputs the policy
now produces, so its errors there are smaller.

**Rubric:** 0.2 for "β penalizes distance from SFT → stops at a smaller KL, near the peak" · 0.15 for "β
limits exploitation but does not fix the proxy; monitor and stop early" · 0.25 for DeepSeekMath: refit the
RM on the new policy's samples and iterate.

**中文重點：** 接 (a)：β 是「離 SFT 越遠罰越多」，β 越大，停在 KL 越小的地方（靠近 KL 10 的高峰）。但 β 沒有修好 reward model，還是要看真正的指標、提早停。DeepSeekMath 的做法：卡住時，**用目前 policy 產生的新樣本重新訓練 reward model**，再繼續跑 GRPO（迭代）。

### D3 (a) Naive vs strict (0.5)

The two training rewards are measured with **different rulers**: each run's own checker. 0.75 from the
naive checker and 0.58 from the strict one cannot be compared. Compare both on the **same** held-out
measure: strict accuracy **19% vs 63%**, and naive is barely above the base model's 15%. The naive policy
most likely learned to **write many numbers**: if the gold number appears anywhere, it gets the reward
(in the lecture run, the number of numbers in the text went from 12 to 26). It hacked a **weak verifier**.
This is the same lesson as A7: the reward is what the checker or judge actually tests, not what you meant.
The A7 judge rewarded made-up venue types and template text that the rubric did not forbid. Fix it with a
stricter checker (an exact final-answer format, as in Tulu 3's RLVR) or a stricter rubric.

**Rubric:** 0.2 for "different rulers; compare on the same held-out checker: 19% vs 63%" · 0.15 for "naive
learned to put many numbers in the answer → hacking a weak verifier" · 0.15 for the link to A7 (the reward
is what the judge or rubric tests; the policy exploits gaps).

**中文重點：** 兩個訓練 reward 是用**兩把不同的尺**量的，不能比；要用同一把尺（held-out strict）：19% vs 63%。naive 學到的是「答案裡塞一堆數字」（投影片：從 12 個數字變成 26 個），這是在鑽弱 verifier 的漏洞。跟 A7 一樣：reward 只是 judge 或 rubric 實際檢查的東西，不是你心裡想的東西。

### D3 (b) R1-Zero, the cold start, and small models (0.5)

R1-Zero's two rule rewards: an **accuracy** reward (is the final answer correct, by a rule checker) and a
**format** reward (the reasoning is written between the thinking tags). It was trained with GRPO from the
base model, with no SFT. R1-Zero's reasoning was hard to read and **mixed languages**. R1 therefore starts
with a **cold-start SFT** on thousands of readable long chain-of-thought examples. This gives readable
output and a better starting point, so the RL is more stable. (Then: reasoning RL with a language-consistency
reward, rejection-sampling SFT on about 800k samples, and a final RL stage for all scenarios.)

For a small model, the two options:
1. **RL it directly:** expensive, and a small model rarely finds long reasoning on its own.
2. **Distill:** SFT on R1's outputs. Cheap and stable, but capped by the teacher, and it needs the teacher.

The paper found that **distillation beat RL** for small models. Trade-off: distillation is cheaper and
better for small models but cannot exceed the teacher. RL can discover new behaviour but needs a capable
model and a lot of compute.

**Rubric:** 0.15 for accuracy + format rewards · 0.15 for the cold start fixing readability and language
mixing and stabilizing RL · 0.2 for small models: distillation beats RL, with the trade-off.

**中文重點：** R1-Zero 用兩個規則獎勵：答案對不對 + 格式對不對（思考寫在 tag 裡），直接從 base 跑 GRPO。問題是寫出來的東西難讀、還會混雜語言，所以 R1 先做 cold-start SFT。小模型：直接跑 RL 很貴又學不起來，**蒸餾 R1 的資料**比較好，但上限就是老師。

---

## Concept coverage (A7 README + L19)

| concept | where |
|---|---|
| Group sampling at temperature; multinomial vs argmax; EOT stops | B1(iii), A1 |
| Padding, mask, teacher-forced log-probs, logits shifted by one | **A1 (i)(ii)(iii)** |
| Causal mask makes right padding harmless (vs A6's left padding) | A1(iii) |
| Group advantage; sign; shift/scale invariance; flat groups | **B1 (i)(ii)(iii)**, C2(a) |
| Loss = −Â log π + β k3; per-token mean, then group mean; only KL left when flat | B1(iii), D1 |
| k3 estimator: values, never negative, vs k1 | **D1 (i)(ii)** |
| PPO ratio = 1 with one update; clip when reusing samples | A3(a) |
| Judge Part 1: order noise, intruder vs chance, humans' mean as ceiling | A2(a) |
| Order noise vs the spread inside a group → noisy advantages; fixes and costs | A2(b), B1(ii) |
| Eval columns: judge CI, rank of 8 (4.5), BLEU, slots, tokens, FineWeb (forgetting) | B2(a) |
| Judge agrees with unseen metrics only where the rubric asks for the same thing | B2(b), B3(a) |
| Reward hacking, repetition, overoptimization, Gao's curve | B2(b), D2(a) |
| KL leash β: what it does; why leash ≈ noleash in 100 steps | D1(iii), D2(b) |
| The rubric is the reward; paraphrase vs naming the behaviour; checking it | B3(a) |
| Judge vs program checker (verifiable reward); time per step; generation dominates | B3(b), A3(a) |
| Two engines: colocated vs disaggregated; async → off-policy; frameworks | A3(b) |
| SFT ceiling; InstructGPT's three steps; 1.3B vs 175B | C3(a) |
| Bradley–Terry; difference only | C2(a) |
| KL-regularized objective | D2(b), C2(b) |
| Four models in PPO; bytes (12 B trained, 2 B frozen); what still is missing | C1 (i)(ii) |
| Value model vs group-mean baseline; GRPO's cost | C1(iii) |
| DPO: implicit reward, Z(x) cancels, offline, no sampling | C2(b) |
| DPO family: KTO (unpaired), IPO, ORPO, SimPO | C3(b) |
| DeepSeekMath iterative RL (refit the RM) | D2(b) |
| Verifiable rewards, weak verifier hacked, Tulu 3 RLVR | D3(a) |
| R1-Zero rule rewards; R1's four stages; distillation vs RL | D3(b) |
