# 引文驗證閘：讓 RAG 的「有憑有據」可被程式檢查

檢索增強生成（RAG）最常見的失效不是檢索不到，是**生成端看起來有憑有據、實際上沒有**——引了不存在的條號，或把條文改寫幾個字再加上引號。這個專案把防線建在程式端：模型輸出的每一條引文都要通過逐字驗證，任一條沒過，整個回答判拒答。

> **In brief** — A Chinese-language legal RAG pipeline whose answers are gated by a deterministic
> citation verifier: every quote the model emits must be a verbatim substring of an article that
> was actually retrieved in that turn. One bad citation rejects the whole answer. The model is
> treated as an untrusted component; the gate is the trusted one.

## 為什麼閘門要建在程式端

「請模型自我檢查有沒有幻覺」是把待驗證的宣稱交給待驗證的對象。這裡三道檢查全部確定性、零 LLM 參與：

| 檢查 | 擋掉什麼 |
|---|---|
| `RETRIEVED` | 引用的條文必須在**這一輪檢索到的集合**內。引到庫裡有、但這輪沒檢索到的條文，等於憑訓練記憶作答，一樣擋——這比「條號存在嗎」嚴格 |
| `VERBATIM` | 引文去空白後必須是該條全文的逐字子字串。**改寫即擋**——意思對但字不對，是最危險的一類，因為它看起來像引文 |
| `NONEMPTY` | 引文至少 8 個字，防「引一個字也算逐字」的空洞引用 |

任一條引文沒過 → 整個回答 `REJECTED`，使用者看到的是拒答而不是答案。**寧可拒答，不可背書。**

## 三層分開量測

RAG 出錯時，「檢索沒撈到」與「模型亂編」必須各自可歸因，混成一個總分會讓改進方向失焦。`src/evaluate.py` 對人工基準集分層量測：

| 指標 | 值 | 說明 |
|---|---:|---|
| `retrieval_recall@5` | 1.000 | 可答題（12）的正解條號都進了 top-5 |
| `answered_rate` | 1.000 | 可答題全數通過引文閘並給出答案 |
| `citation_precision` | 1.000 | 輸出的引文全數逐字驗證通過 |
| `false_answer_rate` | **0.000** | 不可答題（8）全數拒答 |

（qwen3.5:9b／top_k=5／20 題／約 12 分鐘；`golden/results.json` 為逐題原始輸出）

**`false_answer_rate` 是幻覺的直接量測**：答案不在語料裡的題目，系統若給出答案就是幻覺，目標為 0。拒答不算失敗。

### 這組數字看起來太漂亮，所以要說清楚它們證明了什麼

四項滿分的第一版基準集其實**題目太好過**：不可答的題全是跨法域（勞基法、稅法），語意離語料太遠，拒答毫無難度。於是補了四題**同領域近似題**——存款保險理賠上限、銀行最低資本額、通貨交易申報門檻、住宅放款利率上限。這些題的共同點是：**檢索一定會撈到看起來對的條文**（銀行法第46條確實在講存款保險、第23條確實在講最低資本額），但真正的數字在子法或他法，模型最容易在這裡自己把數字補完。實測四題全數拒答，檢索也確實撈出了那些誘餌條文。

更重要的一點誠實揭露：**這 8 題不可答題是模型自己選擇拒答的，閘門根本沒上場**。「沒出事」不等於「防線有效」，兩件事必須分開證明——所以有下面的注入演練。

基準集（`golden/questions.json`）的出題方式：**先讀語料實際條文再寫問題，不憑記憶**——憑記憶出題會把出題者自己的幻覺當成標準答案。

## 證明閘門真的會擋：幻覺注入演練

`src/hallucination_drill.py` 跳過模型，直接餵四種典型幻覺給閘門（檢索仍走真實 pgvector）：

```
✓ ① 條號憑空捏造（庫裡根本沒有第9999條）      → REJECTED／UNRETRIEVED
✓ ② 條號真實但這輪沒檢索到——憑訓練記憶作答    → REJECTED／UNRETRIEVED
✓ ③ 引文改寫：意思對、字不對                  → REJECTED／QUOTE_MISMATCH
✓ ④ 張冠李戴：引文逐字為真，卻掛在別條名下     → REJECTED／QUOTE_MISMATCH
✓ ⑤ 對照組：條號正確且引文逐字                → PASS／VALID
```

單元測試裡另有兩個**探針測試**確認閘門沒有被寫鬆：把「必須命中該條」放寬成「在任一檢索結果裡找得到即可」，三個測試會立刻翻紅（實測驗證過）。守門機制自己也需要被守門。

## 完整重現

```bash
docker compose up -d                       # pgvector
ollama pull bge-m3 && ollama pull qwen3.5:9b
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

./.venv/bin/python src/fetch_corpus.py     # 三部法規 → data/corpus（已附，可跳過）
./.venv/bin/python src/ingest.py           # embedding → pgvector（UPSERT 冪等，重跑筆數不變）
./.venv/bin/python src/evaluate.py            # 基準集評估（20 題，約 12 分鐘）
./.venv/bin/python src/hallucination_drill.py # 幻覺注入演練（不呼叫模型，數秒）
./.venv/bin/python src/rag.py "銀行可以經營哪些業務？"
./.venv/bin/python -m pytest tests/ -q        # 引文閘單元測試（11 則）
```

整條管線**不呼叫任何雲端 API**：embedding 走本機 bge-m3、生成走本機 qwen3.5:9b，成本為零、可離線重跑。代價是生成品質受限於本機小模型（見「誠實邊界」）。

## 語料：為什麼選法規

引文驗證需要一個「可逐字比對的最小單位」，法條的條號＋條文正是這個單位的教科書案例。語料為政府公開資訊（全國法規資料庫），三部與金融、個資直接相關的法規：銀行法、個人資料保護法、洗錢防制法。

## 踩過的坑

**解析器靜默漏資料，而且不會報錯。** 條文行的 class 有兩種寫法（`line-0000` 與 `line-0000 show-number`），第一版 regex 寫死 `line-\d+"`，帶 `show-number` 的行整段抓不到——**語料少了 100 條（180 → 281），程式一次都沒出錯**。抓到它的不是測試，是人工抽讀語料時發現「銀行法第 32 條怎麼不見了」。

修法不只是改 regex，還加了**條數守衛**：頁面上有幾個條號錨點，就該解出幾條（扣除已刪除條文），差太多直接中止抓取。理由是引文閘只能驗「引文是否忠於語料」，**語料本身缺角它驗不出來**——閘門再嚴，也擋不住一個從源頭就殘缺的知識庫。

**模型的引文格式會走鐘，而「修格式」與「放寬驗證」只有一線之隔。** 首輪評估有兩題被擋，查下去發現模型把條號寫進 `law` 欄（`"銀行法 第32條"`）、`label` 給的是「第1項」——項是條之下的層級，不是條。這不是幻覺，是輸出格式問題。修法是在驗證**之前**把條號抽出來（容忍格式），VERBATIM 與 RETRIEVED 兩道檢查一字未改；同時補上兩個探針測試，確保沒有順手把「必須命中該條」鬆成「找得到就算」。分不清這兩者，就會用「提升 answered_rate」的名義把防線拆掉。

**thinking 模式讓同一題從 1.5 秒變成 517 秒。** Ollama 的 qwen3.5 預設開 thinking，first run 直接吃掉 600 秒 timeout。本管線的正確性來自引文閘而非模型推理深度，思考鏈是純成本，`think: False` 關掉。

## 誠實邊界

- 生成模型是**本機 qwen3.5:9b**，不是前沿模型。它偏保守，這對 `false_answer_rate = 0` 有實質貢獻——**這個數字不能全記在閘門頭上**，閘門的效果請看注入演練與單元測試。
- 基準集只有 **20 題**，樣本小到不足以宣稱百分比的精確度；它能支持的結論是「這四類幻覺在這組題目上被擋住了」，不是「幻覺率低於 X%」。
- 可答題的問題是讀著條文寫出來的，**用字與條文天然接近**，`retrieval_recall@5 = 1.0` 因此被高估；真實使用者不會用法條的詞彙提問。這是出題方式的固有偏差，已知但未修正。
- 檢索是**單純的向量相似度 top-k，沒有 rerank、沒有 query rewriting、沒有 hybrid BM25**。`retrieval_recall@k` 是這條管線的天花板，也是最該先改的一環。
- 切塊策略是「一條一塊」，靠法規天然結構省掉 chunking 調參——換成合約、論文這類無明確條號的語料，chunking 與引文定位都要重做。
- 語料為靜態快照，非即時同步法規異動；本專案為技術示範，**不構成法律意見**。
