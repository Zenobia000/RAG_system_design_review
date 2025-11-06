一句話結論：**把 RAG 當成「資料治理 × 檢索工程 × 生成控制」的三相機：先把企業知識變乾淨、檢索變可靠、生成可量測，RAG 才會穩、才會省、才會落地。**

# 《RAG 全實戰攻略》——企業導入版（大學教科書式課綱）

> 對象：已具備 GenAI/LLM 基礎的工程/資料/SRE/產品人員，目標為企業內部知識庫導入與上線。
> 形式：12 週（每週 3–4 小時）或 6 天密集營；每章含講義＋白板推導＋程式實作＋企業實戰作業。
> 產出：一條可上線的**企業內部 RAG 流水線**（含資料治理、索引、混合檢索、重排序、生成控制、評測監控與治理）。

---

## 0. 課程地圖與能力框架（導論）

**學習目標**

* 用「三相機」心智模型：**資料治理（Data Governance）／檢索工程（Retrieval Eng）／生成控制（Gen Control）**拆解 RAG。
* 明確企業成功條件：資料權限邊界、冷啟策略、成本與延遲 SLO。

**重點提綱**

* RAG 與微調（SFT/LoRA）的決策邊界
* 企業導入的三座大山：**資料骯髒、權限複雜、量測失靈**

---

## 1. 企業知識庫治理與文件工程（DocOps）

**學習目標**

* 把雜亂的內部文件做成**檔級—段級—句級**可追溯資料。
* 設計**Chunking/分段＋元資料**標準與稽核流程。

**內容大綱**

* 檔案生命週期：擷取→正規化→敏感資訊處理（PII/PHI/機密等級）→增量更新
* 分段策略：固定長度、語義切分、**樹化摘要/階層式索引（RAPTOR）**、圖譜化（GraphRAG） ([GitHub][1])
* 權限與多租：索引分片、role-based filtering、時間有效性（freshness）

**實作**

* 建立 Doc ETL：PDF/Slides/Confluence/SharePoint→清洗→版面結構擷取（表格、程式碼塊、清單）
* 產出：**企業文件分段與標註規格（Spec）**＋自動化清洗程序（含單元測試）

---

## 2. 索引與向量資料庫選型（Hybrid Retrieval 核心）

**學習目標**

* 做出**Hybrid（稀疏 BM25/SPLADE × 稠密 FAISS/pgvector/Milvus）**的可觀測索引。
* 知道何時用**ColBERTv2**／重排序器（cross-encoder）。

**內容大綱**

* 稠密向量：FAISS / pgvector / Milvus 的差異（佈署、ANN、壓縮、可用性） ([ACL 研討會資料庫][2])
* 稀疏檢索：BM25 與 **SPLADE** 的語義擴展優勢（長尾字詞可解） ([TruLens][3])
* **ColBERTv2** 局部交互檢索與延遲調優、**Contriever** 無監督稠密檢索基線 ([LangChain Docs][4])
* **HyDE**（虛構文件提示）與 **Rerankers**（e5-rerank、bge-rerank 等）在低召回語境的增益

**實作**

* 建立 Hybrid pipeline：BM25/SPLADE + dense，top-k→cross-encoder rerank
* 產出：**檢索卡（Retrieval Cards）**：記錄索引版本、參數、語料快照、延遲與成本

---

## 3. 提示工程到**檢索工程**（Query → Context → Answer）

**學習目標**

* 從「Prompt 錯覺」轉向「Query/Context 工程」，可控地生成**可引用**答案。
* 掌握多輪檢索、查缺補漏與**自我反思路線（Self-RAG/CRAG）**。

**內容大綱**

* Query Reformulation：多樣化查詢、語境展開、反向提示
* **Self-RAG**：生成—檢索—批判—修正迴圈；**CRAG**：錯誤偵測與糾偏策略（低置信度觸發再檢索） ([arXiv][5])
* **GraphRAG**：從局部片段到全域知識（社群偵測＋摘要）應對跨文件議題

**實作**

* 策略路由器：信心/覆蓋率門檻→二次檢索或工具調用
* 產出：**檢索策略藍圖**（何時 HyDE、何時多 hop、何時全域圖）

---

## 4. 生成控制與可引用答案（ grounded generation ）

**學習目標**

* 讓答案**可追溯**（來源段落＋高亮），降低幻覺與法遵風險。
* 熟悉**段落拼接／citation span alignment／答案去重**。

**內容大綱**

* 證據對齊（source attribution）、片段重疊與反重複
* 模板化輸出（JSON/Markdown/表格）與**可引用段落**標示
* 長輸出策略：分段草稿→逐段驗證→合併

**實作**

* 建立**引用對齊器**（對答案句尋找最相近片段，計算支持分數）
* 產出：**企業標準回答模板**（含 citation 與 disclaimer）

---

## 5. RAG 評測與監控（離線評測 × 線上 A/B）

**學習目標**

* 建**可重現**的評測集：**查詢、可接受答案、多來源證據**。
* 熟悉現成框架：**RAGAS、LangSmith、TruLens、RAGBench**。

**內容大綱**

* 指標族：**Context Precision/Recall、Faithfulness、Answer Correctness、Retrieval Recall、Latency/Cost**
* 工具生態：**RAGAS** 指標與用法、**LangSmith** 追蹤、**TruLens** 斷言式評估、**RAGBench** 基準資料集 ([arXiv][6])
* 幻覺評測研究脈絡與最新綜述（HaluEval/Wild、LLM 評測綜述） ([OpenReview][7])

**實作**

* 以公司真實問答建立 **golden set**＋自動化週期性再測
* 產出：**RAG 評測報告**（含風險看板與迭代建議）

---

## 6. 安全、權限與合規（Enterprise-grade）

**學習目標**

* 設計**權限前置過濾（pre-filter）**與審計軌跡。
* 對齊公司資安與法遵：PII/PHI、資料駐留、存取稽核。

**內容大綱**

* 檢索層 RBAC/ABAC、索引分區與租戶隔離
* 敏感資訊遮罩：Prompt-time/Context-time/Answer-time 防護
* 模型輸出審核與 DLP，及**人類在迴路（HITL）**复核流程

**實作**

* 權限測試集（不同角色查同題），驗證資料外洩風險
* 產出：**RAG 安全導入清單與稽核腳本**

---

## 7. 進階變形與「彎道超車」策略

**學習目標**

* 掌握能對**長鏈推理／跨文件議題／隱含關係**明顯加分的方法。

**內容大綱**

* **GraphRAG**（社群/子圖→全域摘要）應對非結構大知識庫
* **RAPTOR**（樹化檢索/階層摘要）處理長文與主題抽象化
* 代理式 RAG（toolformer/計算器/DB/BI 查詢）、工作流編排（LangGraph/LlamaIndex Agents）

**實作**

* 在原管線上掛「圖譜路徑」與「樹化路徑」兩種旁路，比對成本/品質/延遲
* 產出：**多路徑路由策略**（依題型自動選擇 Graph/Tree/Standard）

---

## 8. 成本與延遲工程（Perf/Cost）

**學習目標**

* 在 SLO（p95 延遲、QPS、成本/查詢）下做**批次化、快取、剪枝**。

**內容大綱**

* Index-time 與 Query-time 最佳化：量化、HNSW/IVF 參數、候選集大小
* rerank 階段的**Early-exit**與**兩階段重排序**
* 策略快取（query/result/context）、資料新鮮度與重建頻率

**實作**

* **壓測劇本**與儀表板：延遲分位數、token 成本、檢索召回
* 產出：**SLO 佐證報告**（含預算模型）

---

## 9. 端到端專題（企業題庫落地）

**題型樣例**

1. 技術支援 KB（產品/故障/工單）
2. 研發內網（設計規範、ADR、RFC、郵件會議錄）
3. SOP/製造（工站/參數/偏差處置）

**交付要求**

* 企業資料治理管線（含敏感資訊處理）
* Hybrid 索引＋重排序＋引用對齊
* 評測集（≥200 問，標註可接受答案與引用）
* 監控面板＋SLO 報告＋安全稽核清單

---

## 10. 工具鏈與參考技術（建議組合）

* **向量/檢索**：FAISS、pgvector、Milvus（擴充性/可用性） ([ACL 研討會資料庫][2])
* **稀疏/稠密檢索**：BM25、SPLADE、Contriever、ColBERTv2 ([TruLens][3])
* **工作流**：LangChain(LangGraph)、LlamaIndex（GraphRAG/agents/eval） ([arXiv][8])
* **評測/監控**：RAGAS、LangSmith、TruLens、RAGBench ([arXiv][6])

---

# 評量設計（可量測）

* **作業**（每章）：提交程式與「設計卡」（設計動機、參數、風險）
* **里程碑檢查**：

  * M1：DocOps 管線 + 分段規格
  * M2：Hybrid 檢索 + Rerank p95<300ms（實驗環境）
  * M3：RAGAS/LangSmith 成套評測報告
  * M4：安全與權限 E2E 測試
* **期末專題**：線上 Demo＋SLO/成本/合規報告＋可重現腳本

---

# 參考實作骨架（Py／型別註記精簡版）

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: str
    text: str
    meta: Dict[str, Any]  # {"source":"Confluence","dept":"R&D","acl":["role:SE"],"version":"v3"}

class Retriever:
    def retrieve(self, query: str, k: int = 20) -> List[DocChunk]:
        # hybrid: sparse(bm25/splade) ∪ dense(faiss/pgvector/milvus)
        ...

class Reranker:
    def rerank(self, query: str, chunks: List[DocChunk], top_n: int = 6) -> List[DocChunk]:
        # cross-encoder early-exit + coverage penalty
        ...

class CitationAligner:
    def align(self, answer: str, chunks: List[DocChunk]) -> List[Dict[str, Any]]:
        # map sentences -> supporting spans with scores
        ...

class RAGPipeline:
    def __init__(self, retriever: Retriever, reranker: Reranker, aligner: CitationAligner):
        self.retriever = retriever
        self.reranker = reranker
        self.aligner = aligner

    def answer(self, query: str, user_acl: List[str]) -> Dict[str, Any]:
        # 1) reformulate (HyDE/Self-RAG trigger by confidence) 2) pre-filter by ACL 3) retrieve 4) rerank
        # 5) compose structured prompt 6) generate 7) cite alignment 8) risk flags
        ...
```

---

# 推薦教科書／論文（精選，隨課精讀）

* **GraphRAG**：Microsoft「From Local to Global: A Graph RAG Approach to Query-Focused Summarization」與官方示例。
* **Self-RAG**：Learning to Retrieve, Generate, and Critique（檢索—生成—自我批判）概念。
* **CRAG（Corrective RAG）**：錯誤偵測與自動糾偏工作流。([arXiv][5])
* **RAPTOR**：樹化檢索／階層式摘要以強化長文推理。
* **HyDE**：以假想文件增強零樣本檢索。
* **檢索基礎**：FAISS、pgvector、Milvus 官方文件；ColBERTv2、SPLADE、Contriever。([ACL 研討會資料庫][2])
* **評測實務**：RAGAS、LangSmith、TruLens、RAGBench。([arXiv][6])

---

# 企業導入里程碑（Roadmap／高層溝通版）

| 階段            | 目標               | 核心任務                      | 度量/KPI                      | 風險控管        |
| ------------- | ---------------- | ------------------------- | --------------------------- | ----------- |
| POC (2–4 週)   | 用 1–2 類別文件打通 E2E | DocOps、Hybrid 原型、RAGAS 基線 | Faithfulness ≥0.6、p95<800ms | 垃圾資料→嚴格清洗標準 |
| Pilot (4–8 週) | 擴到 5–10 類別、權限同步  | ACL 前置過濾、監控面板             | 查全率+15%、誤引降 30%             | 權限錯配→自動化測資  |
| Prod (8–12 週) | 正式上線＋SLO         | 快取/批次化、成本看板               | p95<500ms、月成本可預測            | 漏訊新鮮度→增量重建  |
| Scale (持續)    | 多 BU/多租          | 圖譜/樹化旁路、A/B               | 工單解決時長下降 20%                | 變更管理／版本凍結   |

---

## 常見坑位與對策（工程視角）

* **只調 Prompt 不調索引** → 先做 **Retrieval Cards**；記錄召回與重排序差距。
* **權限過濾放生成後** → 一律**前置過濾**，並做權限測試集。
* **沒有評測集** → 先標 200 題**公司題庫**，每題至少兩段可接受證據。
* **只上 dense** → 商業長尾文本需 **Hybrid+Rerank**；必要時加 **HyDE/GraphRAG**。
* **延遲爆炸** → two-stage 重排序＋early-exit＋策略快取。

---

## 課後延伸（選修模組）

* **結構化資料 RAG**：SQL/BI 工具代理與可驗證答案
* **事件新鮮度**：Fresh/Time-aware Retrievers、新聞/FAQ 滾動更新
* **跨語言 RAG**：中英混合語料與詞彙同義映射

---

# 心法內化（像 5 歲小孩也懂）

把 RAG 想成**找資料、拼圖、畫重點**：
先把資料放整齊（不髒不亂）→ 找到對的拼圖（好檢索）→ 畫上重點和出處（有證據的答案）。做得好，大家就放心用。

# 口訣記憶（3 點）

1. **先清再搜再說**（清資料→搜文件→再說話）。
2. **混檢索配重排**（BM25/SPLADE＋向量＋Rerank）。
3. **量測才上線**（RAGAS/LangSmith/TruLens 指標綁 SLO）。

> 需要我把這套課綱轉成**教學投影片大綱＋每章實作指引（含評分規準與 Git 範本）**嗎？我可以直接附上模板與示例專題清單。

[1]: https://github.com/microsoft/graphrag?utm_source=chatgpt.com "microsoft/graphrag: A modular graph-based Retrieval ..."
[2]: https://aclanthology.org/2023.acl-long.99/?utm_source=chatgpt.com "Precise Zero-Shot Dense Retrieval without Relevance Labels"
[3]: https://www.trulens.org/getting_started/core_concepts/rag_triad/?utm_source=chatgpt.com "RAG Triad"
[4]: https://docs.langchain.com/langsmith/evaluate-rag-tutorial?utm_source=chatgpt.com "Evaluate a RAG application - Docs by LangChain"
[5]: https://arxiv.org/html/2501.00309v2?utm_source=chatgpt.com "Retrieval-Augmented Generation with Graphs (GraphRAG)"
[6]: https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
[7]: https://openreview.net/pdf?id=sjwX4Vif03&utm_source=chatgpt.com "Evaluating Hallucinations of Language Models in the Wild"
[8]: https://arxiv.org/abs/2212.10496?utm_source=chatgpt.com "Precise Zero-Shot Dense Retrieval without Relevance Labels"
