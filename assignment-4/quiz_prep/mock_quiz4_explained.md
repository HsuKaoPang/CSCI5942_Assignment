# Mock Quiz 4：逐題詳解

這份詳解跟 `mock_quiz4_answers.md` 搭配使用。answers 是**考試要寫的答案**，這份是**為什麼答案是這樣**。
每一題都分成四段：考什麼 → 推理過程 → 常見錯誤 → 考試時的一句話。

---

# Set A（1.0B 模型、1 個節點、4 張 A40）

## 1(a) naive DP 多出來的 0.5 秒，有多少是 ring all-reduce 造成的？

**考什麼**：會不會算 DP 每一步的通訊量和通訊時間，再拿去跟實際量到的時間比較（對應 Part 1 Q1）。

**推理過程**

1. **完美擴展的時間**：1 張 GPU 要 8.0 秒，4 張就是 8.0 ÷ 4 = **2.0 秒**。實際的 naive 是 2.5 秒，所以額外開銷是 **0.5 秒**。
2. **要傳多少資料**：DP 每一步要對所有梯度做 all-reduce。
   - 參數 1.0B 個，梯度是 fp32，每個 4 bytes → **4 GB**。
3. **ring all-reduce 實際傳多少**：ring 分成兩個階段，每個 rank 都要傳大約 2 份資料的量：
   - reduce-scatter：傳 (P−1)/P 份
   - all-gather：再傳 (P−1)/P 份
   - 合計是 **2(P−1)/P**。P=4 時是 2 × 3/4 = **1.5**，所以每個 rank 要傳 1.5 × 4 = **6 GB**。
4. **需要多久**：6 GB ÷ 16 GB/s = **0.375 秒**。
5. **比較**：0.375 ÷ 0.5 = **75%**。理論解釋了四分之三的額外開銷。
6. **剩下的 0.125 秒是什麼**：
   - 梯度是**一個參數一個參數**分別做 all-reduce 的，會有幾百次。
   - 很多參數很小，例如 bias 和 LayerNorm。看表上的數字：**1 MB 的 buffer 只有 6 GB/s**，比 16 GB/s 慢很多，因為小資料量時是被延遲卡住，而不是被頻寬卡住。
   - 每次呼叫還有啟動和同步的成本。

**常見錯誤**
- 忘記乘以 **1.5**，直接用 4 GB ÷ 16 = 0.25 秒。
- 用 bf16（2 bytes）來算。梯度是 **fp32**，題目有寫明。
- 只寫「因為有通訊」，沒有算出數字。這樣會失去「用數字支持結論」那一部分的分數。

**一句話**：Gradients are 4 GB; a ring puts 1.5 × 4 = 6 GB per link, 6/16 = 0.375 s ≈ 75% of the 0.5 s gap; the rest is per-parameter calls on small, latency-bound buffers.

---

## 1(b) 換成 4 個節點、每個節點 1 張 GPU，頻寬只有 4 GB/s，overlap 能藏多少？

**考什麼**：換情境的應用題。overlap 的**上限**是由什麼決定的（對應 Part 1 Q2）。

**推理過程**

1. **新的通訊時間**：一樣是 6 GB，但頻寬變成 4 GB/s → 6 ÷ 4 = **1.5 秒**。
2. **naive 的預測**：2.0 + 1.5 ≈ **3.5 秒**。
3. **overlap 能藏多少？** 要先想清楚「**有多少計算時間可以拿來掩蓋通訊**」：
   - 每個 rank 的 micro-batch 數 = 128 ÷ (4 張 GPU × micro 4) = **8 個**。
   - 因為 `no_sync`，前 7 個 micro-batch 不會觸發通訊，**只有第 8 個的 backward 會**。
   - 每個 micro-batch 的計算時間 = 2.0 ÷ 8 = 0.25 秒，其中 backward 佔 2/3 → **約 0.17 秒**。
   - 所以 overlap **最多只能藏 0.17 秒**，剩下的 1.33 秒一定會露出來。
4. **還有一個限制**：第一層的梯度是最後才算出來的，它開始傳的時候 backward 已經結束了，沒有任何計算可以掩蓋它。
5. **結論**：step 時間至少是 2.0 + 1.33 ≈ **3.3 秒**，擴展效率只剩 8 ÷ (4 × 3.3) ≈ **60%**。隊友說的「overlap 會藏掉大部分」是**錯的**。

**常見錯誤**
- 以為 overlap 可以藏住**整個** backward（2.0 × 2/3 = 1.33 秒）。這樣就忽略了梯度累積：只有最後一個 micro-batch 的 backward 能用來掩蓋。
- 只說「藏不住」，沒有說出**上限是由什麼決定的**。

**一句話**：Comm becomes 6/4 = 1.5 s; overlap can only hide behind backward still running, and with no_sync only the last of 8 micro-batches (~0.17 s of backward) triggers reductions, so the step stays ≥ 3.3 s.

---

## 2(a) TP=4 比 1 張 GPU 還慢，是不是 code 寫壞了？

**考什麼**：TP 的成本和好處各是什麼（對應 Part 2 Q1）。

**推理過程**

1. **先排除「code 壞了」**：TP=4 和 DP=4 的**計算量完全一樣**，都是把同樣的工作分給 4 張 GPU。所以 9.1 − 2.2 = **6.9 秒**的差距**一定是通訊造成的**，不是計算。
2. **TP 的通訊為什麼這麼貴**：
   - **傳的是 activation，不是梯度**。每一層的 attention 和 MLP 各需要一次 forward 的 all-reduce（`f_bar`）和一次 backward 的 all-reduce（`f`），所以**每層 4 次，而且每個 micro-batch 都要做**。
   - **卡在 critical path 上**：下一層要用 all-reduce 的結果當輸入，GPU 只能停下來等。
   - 相比之下，DP **每一步只傳一次梯度**，而且可以跟 backward 同時進行。
3. **TP 的好處**：**記憶體**。7 GiB 對 20 GiB，因為權重、梯度和 Adam 狀態都被切成 4 份。
4. **該選哪一個**：這個模型只要 20 GiB，一張 44 GiB 的 A40 就放得下，所以**選 DP=4**。TP 只有在模型放不下的時候才值得用。

**常見錯誤**
- 說「TP 的計算比較多」。不對，計算量是一樣的。
- 只說 TP 不好，沒有說出它的**好處是省記憶體**。評分標準一定會有「What is the benefit of TP」這一項。
- 沒有給出明確的選擇。題目問「Which would you ship」，就一定要回答。

**一句話**：Same FLOPs, so the ~7 s gap is TP's activation all-reduces, four per block per micro-batch and on the critical path, versus DP's single overlappable gradient reduce; TP's payoff is memory (7 vs. 20 GiB), but the model fits, so ship DP=4.

---

## 2(b) 預測 PP 的 bubble，並解釋 AFAB 和 1F1B 的記憶體差異

**考什麼**：bubble 公式、用 bubble 回推 step 時間，以及兩種排程在記憶體上的差異（對應 Part 3）。

**推理過程**

1. **bubble 公式**：(p−1)/(m+p−1)，p=4
   - m=4：3/7 ≈ **43%**
   - m=16：3/19 ≈ **16%**
2. **用 bubble 回推 step 時間**：
   - 如果沒有 bubble，每個 stage 負責 1/4 的計算 → 8.0 ÷ 4 = **2.0 秒**。
   - 有 bubble 時，只有 (1 − bubble) 的時間在做事 → step = 2.0 ÷ (1 − bubble)。
   - m=4：2.0 ÷ (4/7) = **3.5 秒**，實際是 3.6 秒 ✓
   - m=16：2.0 ÷ (16/19) ≈ **2.4 秒**，實際是 2.6 秒。多出的 0.2 秒來自**stage 之間的傳輸**，以及**最後一個 stage 比較重**（它還要算 LM head 和 loss）。
3. **為什麼 AFAB 和 1F1B 一樣快**：bubble 只取決於 p 和 m，跟 forward/backward 的**執行順序無關**。所以兩者都是 2.6 秒。
4. **為什麼記憶體差這麼多**：
   - **AFAB**：所有 forward 都做完才開始 backward，所以 **16 個 micro-batch 的 activation 會同時存在**。
   - **1F1B**：做一個 forward、就做一個 backward，同時存在的 activation **最多 p=4 份**。
   - 所以是 18 GiB 對 7 GiB。

**常見錯誤**
- 以為 1F1B 會比較快。**1F1B 只省記憶體，不會減少 bubble。** L07 投影片寫的是「doesn't reduce pipeline bubbles」。
- 只寫公式，沒有回推 step 時間來驗證。題目明確要你「check whether the step times agree」。

**一句話**：Bubble = 3/7 ≈ 43% and 3/19 ≈ 16%, giving 2.0/(1−b) ≈ 3.5 s and 2.4 s, matching 3.6 and 2.6 s; the order doesn't change the bubble, but AFAB holds all 16 micro-batches while 1F1B holds at most p = 4.

---

# Set B（4.0B 模型、2 個節點、8 張 A40）

## 1(a) 為什麼 DP=8 會 OOM，而 DP4×TP2 放得下？

**考什麼**：記憶體的計算方式，以及每一種平行方式在記憶體上的效果（對應 Part 4）。

**推理過程**

1. **每個參數要幾 bytes**（**一定要背**）：
   | 項目 | bytes |
   |---|---|
   | 權重（fp32） | 4 |
   | 梯度（fp32） | 4 |
   | Adam 的兩個 moment（fp32 × 2） | 8 |
   | **合計** | **16** |
2. **模型狀態總量**：4.0B × 16 = **64 GB**（約 60 GiB）。
3. **DP=8**：DP 不會切模型，**每張 GPU 都要放完整的 64 GB**，超過 44 GiB，所以在算 activation 之前就 OOM 了。
4. **DP4×TP2**：TP 把模型切成 2 份 → 64 ÷ 2 = **32 GB**，再加上 activation，大約是實際量到的 **38 GiB**，放得下。
5. **每一種平行方式買到什麼**：
   - **DP**：只增加吞吐量，**完全不省記憶體**。
   - **TP**：每張 GPU 的模型狀態 ÷ tp，部分 activation 也會被切開。
   - **PP**：每張 GPU 的模型狀態 ÷ pp，但每個 stage 要存最多 p 個 micro-batch 的 activation。
   - 綜合起來，**每張 GPU 的模型狀態 ≈ 總量 ÷ (tp × pp)**。

**常見錯誤**
- 以為 DP=8 會把記憶體分成 8 份。**DP 不會切模型。**（ZeRO 才會，見 2(a)。）
- 只算權重 4 bytes，忘記梯度和 Adam。

**一句話**：Model state is 4.0B × 16 B = 64 GB; DP replicates it, so every DP=8 rank needs 64 GB > 44 GiB, while TP=2 halves it to 32 GB (38 GiB with activations). DP buys throughput, TP and PP buy memory.

---

## 1(b) TP=8 最省記憶體，卻比 TP=4 慢這麼多，為什麼？

**考什麼**：硬體的拓撲：節點內和跨節點的連線差異，以及 TP 應該放在哪裡。

**推理過程**

1. **一個節點有 4 張 GPU**：
   - TP=4 的 group 是 4 張 GPU，**可以完全放在一個節點內**。
   - TP=8 的 group 是 8 張 GPU，**一定會跨越 2 個節點**。
2. **跨節點的影響**：TP 每一層的 all-reduce 都要經過**節點之間比較慢的網路**，而這些 all-reduce 又都在 critical path 上，每一次 GPU 都要停下來等。
3. **結果**：通訊最頻繁的 TP，偏偏被放在最慢的連線上，所以從 17.5 秒變成 96 秒。
4. **放置規則**：
   - **TP 放在節點內**：通訊最頻繁，需要最快的連線。
   - **PP 跨節點**：每個 micro-batch 只在 stage 交界處傳一次 activation。
   - **DP 也可以跨節點**：每一步只通訊一次，而且可以 overlap。

**常見錯誤**
- 說「TP=8 的通訊量比較大」。其實每一層的 all-reduce 次數是一樣的，**差別在於走的是哪一條連線**。
- 沒有說出放置規則。題目明確要求「State the placement rule」。

**一句話**：A TP=4 group fits in one 4-GPU node, but TP=8 spans both nodes, so every block's critical-path all-reduce crosses the slow inter-node link; keep TP within a node and let PP or DP cross nodes.

---

## 2(a) 用 activation checkpointing 能讓 DP=8 放得下嗎？

**考什麼**：分清楚 activation 記憶體和模型狀態記憶體，以及 ZeRO 的作用（L07 投影片的內容）。

**推理過程**

1. **activation checkpointing 在做什麼**：forward 時**不存** activation，到 backward 需要時再**重新計算**。它是用計算時間換 activation 的記憶體。
2. **為什麼沒用**：DP=8 的問題在於**模型狀態本身就有 64 GB**，連 activation 都還沒開始算就已經超過了。checkpointing 只能減少 activation，**完全沒碰到模型狀態**。
3. **該用什麼**：**ZeRO**（Zero Redundancy Optimizer）。DP 的問題是 8 張 GPU 存了 8 份一模一樣的東西，ZeRO 就是把這些重複的部分切開，每張 GPU 只存其中一份：
   | 階段 | 切開什麼 | 每張 GPU 需要（8 張） |
   |---|---|---|
   | 原本的 DP | 什麼都不切 | 16 + 16 + 32 = 64 GB |
   | ZeRO-1 | Adam 狀態 | 16 + 16 + 32/8 = **36 GB**（加上 activation 很擠） |
   | ZeRO-2 | Adam + 梯度 | 16 + 16/8 + 4 = **22 GB** |
   | ZeRO-3 | 全部 | 64/8 = **8 GB** |
4. **成本**：**通訊變多**。
   - 梯度要改用 reduce-scatter，每張 GPU 只拿到自己負責的那一份。
   - 參數要用 all-gather 把完整的權重拼回來。ZeRO-3 在每次 forward 和 backward 前都要做一次。
   - 這些通訊都會經過跨節點的網路。

**常見錯誤**
- 說 checkpointing「可以省一點記憶體，所以有幫助」。重點是**它省的是不對的那一部分**，必須明確說出不會成功。
- 提到 ZeRO 卻沒有說成本。設計題都要討論取捨。

**一句話**：No: checkpointing only shrinks activations, but the 64 GB of model state alone exceeds 44 GiB. ZeRO shards DP's redundant state (ZeRO-1 Adam → ~36 GB, ZeRO-2 + grads → ~22 GB, ZeRO-3 + params → ~8 GB) at the cost of extra reduce-scatter and all-gather traffic.

---

## 2(b) 實際的 idle 為什麼一直比預測高？1F1B 的記憶體為什麼反而下降？

**考什麼**：找異常的診斷題，用的是你自己 Part 3 的數據。

**推理過程（第一個現象）**

公式 (p−1)/(m+p−1) 有兩個**隱含的假設**：
1. **每個 stage 一樣重**：實際上不是。最後一個 stage 要算 **50304 寬的 LM head 和 cross-entropy**，比其他 stage 重很多，其他 stage 只能等它。
2. **通訊不花時間**：實際上每個 micro-batch 都要在 stage 之間**傳 activation（forward）和梯度（backward）**，這段時間 stage 都在等。

這兩種成本**每個 micro-batch 都要付一次**，所以 m 變大時，它們也跟著變多，差距不會消失（m=32 時仍然是 28% 對 9%）。

**推理過程（第二個現象）**

1. **global batch 固定是 32** → micro = 32 ÷ m。m 從 4 變成 32，micro 就從 8 變成 1。
2. **1F1B** 最多同時存 p 個 micro-batch → activation ∝ p × micro = 4 × 32/m → **m 越大，記憶體越少**。
3. **AFAB** 存全部 m 個 → activation ∝ m × micro = 32，**理論上應該是固定的**。但實際上一直增加，原因是每個 micro-batch 都有額外的開銷，例如 autograd 存的東西和記憶體碎片，而且最後一個 stage 本來就要存很大的 logits，所以 m=32 時就超過 44 GiB 了。
4. **m=4 時兩者一樣**（都是 17.1 GiB）：因為 m = p，1F1B 也要把 4 個全部存著。

**常見錯誤**
- 以為 1F1B 的記憶體應該「固定不變」。它的**份數**最多是 p，但**每一份的大小**會隨著 micro 變小。
- 只寫「因為有通訊」，沒有點出「**stage 不平均**」這一點。

**一句話**：The formula assumes equal stages and free communication; real stages pay per-micro-batch P2P transfers and wait on the LM-head-heavy last stage. With the batch fixed, micro = 32/m shrinks, and 1F1B holds ≤ p of them, so its memory falls, while AFAB holds all m and overflows at m = 32.

---

# 快問快答

## 1. `f` 在 forward 是恆等函數。如果它在 backward 也是恆等函數，會壞掉什麼？

- `f` 位於 TP 區域的**入口**。forward 時，每張 GPU 都拿到**完整的 x**，然後用**自己那一塊** W 來算。
- 所以 backward 時，每張 GPU 只算出了「**x 的梯度的一部分**」，也就是自己那塊 W 貢獻的部分。
- **正確的 x 梯度 = 所有 GPU 的部分加起來**，這就是 `f` 在 backward 時要做 all-reduce 的原因。
- 如果不做加總：x 前面的所有東西，包括 LayerNorm、前面的 block、embedding，都會收到**錯誤的、不完整的梯度**。而且每張 GPU 收到的部分不一樣，所以這些應該一模一樣的權重，在不同 GPU 上會**越來越不同**。
- 你寫 TP 時看到的測試 `test_backward_matches_one_gpu`，抓的就是這個錯誤。

## 2. 為什麼 `RowParallelLinear` 的 bias 要在 all-reduce 之後才加？

- row parallel 的每張 GPU 算出的是**部分和**，all-reduce 會把它們**加總**。
- 如果每張 GPU 在加總之前就加上 bias：(p₀ + b) + (p₁ + b) = p₀ + p₁ + **2b**，bias 就被加了 tp 次。
- 你寫 RowParallelLinear 時的實驗：正確答案是 110，在加總前加 bias 會變成 210。

## 3. 把 `p.grad /= world` 移到送出 async all-reduce **之後**，為什麼測試時好時壞？

- `async_op=True` 只是**送出**請求，all-reduce 會在背景繼續執行，**這段時間它還在讀寫 `p.grad`**。
- 這時候對 `p.grad` 做除法，就會跟背景的 all-reduce **同時修改同一塊記憶體**，這叫做 **race condition**。
- 結果取決於**哪一邊先完成**：有時候除法在 all-reduce 開始讀之前就做完了，結果正確；有時候是在中途，結果就錯了。所以測試才會時好時壞。
- 正確的做法：**在送出之前先除**（我們的寫法），或是**在 `wait()` 之後再除**。

## 4. 1F1B 的穩定期為什麼一定要用 `send_forward_recv_backward`，不能先 blocking send、再 blocking recv？

- 穩定期時，**stage i** 要「把 activation 送給 i+1，並從 i+1 收梯度」，而 **stage i+1** 同時要「把梯度送給 i，並從 i 收 activation」。
- 如果兩邊都是**先 blocking send**：
  - stage i 呼叫 `send`，**等** stage i+1 來 `recv`。
  - stage i+1 也呼叫 `send`，**等** stage i 來 `recv`。
  - **兩邊都在等對方先收**，誰也不會往下走，程式就永遠卡住了。這叫做 **deadlock**。
- `send_forward_recv_backward` 用的是 `batch_isend_irecv`，會**同時**把「送」和「收」這兩件事交出去，不需要等其中一個完成，所以不會卡住。

## 5. 為什麼 `broadcast_module_state` 除了 `parameters()`，還要同步 `buffers()`？

- buffers 是**不會被訓練、但會參與 forward 計算**的 tensor，例如 causal mask、BatchNorm 的 running mean。
- 如果 buffers 在不同 rank 上不一樣，那麼就算 parameters 完全相同，**算出來的輸出也會不同**，接著 loss 和梯度也會不同，各 rank 的模型就會越來越不一樣。
- DP 要保證的是**整個模型**都一樣，不只是會被訓練的那一部分。

## 6. 為什麼跨過兩個節點之間慢速連線的，應該是 PP 的 stage 交界，而不是 TP？

- **TP**：每一層、每個 micro-batch 都要做 4 次 all-reduce，而且都在 critical path 上。
- **PP**：只在 stage 交界處，每個 micro-batch 傳 1 次 activation 和 1 次梯度，而且是 point-to-point，只傳給隔壁的 stage。
- 以 huge（48 層）、pp=2 為例，每個 micro-batch 需要跨節點的次數：**TP 是 192 次，PP 是 2 次**，相差將近 100 倍。
- 所以要讓通訊最少的 PP 走慢的連線，通訊最多的 TP 留在節點內。

## 7. 有梯度累積時，`no_sync()` 為什麼對通訊量很重要？

- 梯度累積時，`p.grad` 會在每個 micro-batch 的 backward 後**一直累加**，直到最後一個才是完整的梯度。
- **如果沒有 `no_sync`**：每個 micro-batch 都會做一次完整的 all-reduce。8 個 micro-batch × 4 GB = **32 GB**，其中 7 次是在傳還不完整的梯度，完全是浪費。
- **有 `no_sync`**：只有最後一個 micro-batch 做 all-reduce → **4 GB**，通訊量變成原本的 **1/8**。
- **結果一樣**：因為加總的順序可以互換，先在本地加完再到 rank 之間加總，跟每次都加總一次，結果完全相同。
- Part 1 的測試 `test_no_sync_changes_nothing_but_the_traffic` 驗證的就是這件事。
