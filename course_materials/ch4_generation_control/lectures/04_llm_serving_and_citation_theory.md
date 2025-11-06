# LLM 服務與引用理論
## 大學教科書 第4章：大型語言模型的生產部署與引用系統

**課程編號**: CS785 - 企業級檢索增強生成系統
**章節**: 第4章 生成控制
**學習時數**: 8小時
**先修課程**: 深度學習基礎, 自然語言生成, 第0-3章
**作者**: 語言模型研究團隊
**最後更新**: 2025-01-06

---

## 📚 學習目標 (Learning Objectives)

完成本章學習後，學生應能夠:

1. **理論基礎**: 掌握大型語言模型的數學原理和生成控制理論
2. **系統架構**: 設計企業級 LLM 服務架構和引用驗證系統
3. **性能優化**: 實現高效能的模型推理和批次處理策略
4. **品質控制**: 建立完整的事實檢查和引用對齊機制

---

## 1. 大型語言模型的理論基礎

### 1.1 Transformer 架構的數學原理

#### **注意力機制的數學表示**

**定義 1.1** (多頭注意力機制): 對於輸入序列 $X \in \mathbb{R}^{n \times d}$，多頭注意力計算為：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每個注意力頭定義為：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**定理 1.1** (注意力複雜度): 標準注意力機制的時間複雜度為 $O(n^2d)$，其中 $n$ 為序列長度，$d$ 為特徵維度。

#### **位置編碼的理論分析**

**RoPE (Rotary Position Embedding)** 的數學原理 (Su et al., 2021)[^23]:

$$f(x_m, m) = \begin{pmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{pmatrix} \begin{pmatrix}
x_{m,2i} \\
x_{m,2i+1}
\end{pmatrix}$$

其中 $\theta = 10000^{-2i/d}$，$m$ 為位置索引。

**性質 1.1** (RoPE 的相對位置不變性): RoPE 編碼保證了相對位置關係在內積空間中的線性表示。

### 1.2 生成控制的概率理論

#### **條件生成的數學框架**

**定義 1.2** (條件文本生成): 給定上下文 $c$ 和查詢 $q$，模型生成回應 $y$ 的概率為：

$$P(y|c,q) = \prod_{t=1}^{|y|} P(y_t|y_{<t}, c, q)$$

**定理 1.2** (生成忠實度界限): 對於檢索增強生成，忠實度的理論上界為：

$$\text{Faithfulness} \leq \min\left(P(\text{relevant}|c), P(\text{faithful}|c,\text{relevant})\right)$$

**證明思路**: 生成忠實度受制於上下文相關性和模型的忠實生成能力，兩者的最小值決定了系統的忠實度上界。□

#### **引用對齊的資訊理論**

**定義 1.3** (引用對齊): 生成文本與源文檔之間的資訊對應關係，量化為：

$$\text{Citation-Alignment} = \frac{I(Y;C)}{H(Y)}$$

其中 $I(Y;C)$ 為生成文本 $Y$ 與上下文 $C$ 的互資訊，$H(Y)$ 為生成文本的熵。

---

## 2. vLLM 生產部署深度解析

### 2.1 vLLM 架構原理

#### **PagedAttention 的創新機制**

vLLM (Kwon et al., 2023)[^24] 的核心創新是 PagedAttention 機制：

**原理**: 將注意力計算的 KV Cache 分頁管理，類似作業系統的虛擬記憶體：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q \cdot \text{PagedK}^T}{\sqrt{d_k}}\right) \cdot \text{PagedV}$$

**優勢分析**:
- **記憶體效率**: 減少 60-80% 記憶體浪費
- **動態批次**: 支援不同序列長度的動態批次處理
- **並行優化**: 更好的 GPU 利用率

#### **企業級 vLLM 部署實現**

```python
import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
from typing import Dict, List, Optional, AsyncGenerator
import time
from dataclasses import dataclass

@dataclass
class GenerationRequest:
    """生成請求數據結構"""
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    metadata: Dict
    priority: int = 5  # 1-10, 10為最高優先級

class EnterprisevLLMService:
    """企業級 vLLM 服務"""

    def __init__(self, model_config: Dict):
        # vLLM 引擎配置
        self.engine_args = AsyncEngineArgs(
            model=model_config["model_path"],
            tokenizer=model_config.get("tokenizer_path"),

            # 並行配置
            tensor_parallel_size=model_config.get("tensor_parallel_size", 4),
            pipeline_parallel_size=model_config.get("pipeline_parallel_size", 1),

            # 記憶體優化
            gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.9),
            swap_space=model_config.get("swap_space", 4),  # GB

            # 性能優化
            max_num_batched_tokens=model_config.get("max_batched_tokens", 8192),
            max_num_seqs=model_config.get("max_num_seqs", 256),
            enable_chunked_prefill=model_config.get("enable_chunked_prefill", True),

            # 精度配置
            dtype=model_config.get("dtype", "bfloat16"),
            quantization=model_config.get("quantization"),  # "awq", "gptq"

            # 其他配置
            disable_log_stats=False,
            trust_remote_code=True
        )

        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

        # 請求調度器
        self.request_scheduler = RequestScheduler()

        # 性能監控
        self.performance_monitor = vLLMPerformanceMonitor()

    async def generate_with_context(self, prompt: str,
                                  context_sources: List[Dict],
                                  generation_config: Dict) -> Dict:
        """帶上下文的受控生成"""

        # 階段1: 上下文預處理
        processed_context = await self._preprocess_context(
            context_sources, generation_config
        )

        # 階段2: 提示工程
        structured_prompt = await self._build_structured_prompt(
            prompt, processed_context, generation_config
        )

        # 階段3: 生成參數配置
        sampling_params = self._configure_sampling_parameters(generation_config)

        # 階段4: 受控生成
        generation_result = await self._controlled_generation(
            structured_prompt, sampling_params
        )

        # 階段5: 後處理與驗證
        validated_result = await self._post_process_and_validate(
            generation_result, context_sources, prompt
        )

        return validated_result

    async def _preprocess_context(self, sources: List[Dict],
                                config: Dict) -> Dict:
        """預處理上下文資料"""

        # 1. 來源排序與選擇
        ranked_sources = await self._rank_sources_by_relevance(sources, config)

        # 2. 內容清理與格式化
        cleaned_sources = []
        for source in ranked_sources[:config.get("max_sources", 10)]:
            cleaned_content = await self._clean_source_content(source["content"])

            # 添加來源標識
            source_id = f"SOURCE_{len(cleaned_sources) + 1}"
            formatted_content = f"[{source_id}] {cleaned_content}"

            cleaned_sources.append({
                "id": source_id,
                "content": formatted_content,
                "metadata": source.get("metadata", {}),
                "confidence": source.get("confidence", 1.0)
            })

        # 3. 上下文長度控制
        total_length = sum(len(s["content"]) for s in cleaned_sources)
        max_context_length = config.get("max_context_tokens", 16384)

        if total_length > max_context_length:
            # 智能截斷：保留最重要的來源
            truncated_sources = await self._intelligent_truncation(
                cleaned_sources, max_context_length
            )
        else:
            truncated_sources = cleaned_sources

        return {
            "formatted_sources": truncated_sources,
            "total_tokens": sum(len(s["content"]) for s in truncated_sources),
            "truncated": len(truncated_sources) < len(cleaned_sources)
        }

    async def _build_structured_prompt(self, user_query: str,
                                     context: Dict,
                                     config: Dict) -> str:
        """構建結構化提示"""

        template_type = config.get("template_type", "standard")

        if template_type == "enterprise_qa":
            template = """
您是一位專業的企業知識助理。請基於提供的權威資料來源回答問題。

## 重要指示
1. 僅基於提供的資料來源回答問題
2. 對所有關鍵陳述使用 [SOURCE_N] 格式引用來源
3. 如果資料不足，明確說明限制
4. 保持客觀中性的語調
5. 提供結構化的回答

## 資料來源
{context_text}

## 用戶問題
{user_query}

## 回答格式
**主要回答**: [基於資料的直接回答]
**詳細說明**: [支持性細節和分析]
**資料來源**: [引用的具體來源]
**限制說明**: [如有資料限制或不確定性]

回答:
"""
        elif template_type == "technical_support":
            template = """
您是技術支援專家。請基於技術文檔提供準確的技術指導。

## 技術資料
{context_text}

## 技術問題
{user_query}

## 回答要求
1. 提供明確的技術解決方案
2. 列出具體操作步驟
3. 標註潛在風險和注意事項
4. 引用相關技術文檔 [SOURCE_N]

技術回答:
"""
        else:  # standard template
            template = """
請基於以下資料回答問題，並確保：
1. 回答準確且基於事實
2. 適當引用資料來源 [SOURCE_N]
3. 承認資料限制

資料:
{context_text}

問題: {user_query}

回答:
"""

        # 格式化模板
        context_text = "\n\n".join([
            source["content"] for source in context["formatted_sources"]
        ])

        structured_prompt = template.format(
            context_text=context_text,
            user_query=user_query
        )

        return structured_prompt

    def _configure_sampling_parameters(self, config: Dict) -> SamplingParams:
        """配置採樣參數"""

        return SamplingParams(
            temperature=config.get("temperature", 0.1),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k", 50),
            max_tokens=config.get("max_tokens", 2048),
            stop=config.get("stop_sequences", ["\n\nHuman:", "<|end|>"]),
            presence_penalty=config.get("presence_penalty", 0.0),
            frequency_penalty=config.get("frequency_penalty", 0.0),
            repetition_penalty=config.get("repetition_penalty", 1.1),
            include_stop_str_in_output=False
        )

    async def _controlled_generation(self, prompt: str,
                                   sampling_params: SamplingParams) -> Dict:
        """執行受控生成"""

        request_id = random_uuid()
        start_time = time.time()

        try:
            # 生成文本
            outputs = []
            async for request_output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                outputs.append(request_output)

            # 獲取最終輸出
            final_output = outputs[-1]
            generated_text = final_output.outputs[0].text

            generation_time = time.time() - start_time

            # 記錄性能指標
            await self.performance_monitor.record_generation(
                request_id=request_id,
                input_tokens=len(prompt.split()),  # 簡化計算
                output_tokens=len(generated_text.split()),
                generation_time=generation_time,
                model_name=self.engine_args.model
            )

            return {
                "generated_text": generated_text.strip(),
                "request_id": request_id,
                "generation_time": generation_time,
                "token_count": {
                    "input": len(prompt.split()),
                    "output": len(generated_text.split()),
                    "total": len(prompt.split()) + len(generated_text.split())
                },
                "finish_reason": final_output.outputs[0].finish_reason
            }

        except Exception as e:
            return {
                "error": str(e),
                "request_id": request_id,
                "generation_time": time.time() - start_time
            }

    async def _post_process_and_validate(self, generation_result: Dict,
                                       context_sources: List[Dict],
                                       original_query: str) -> Dict:
        """後處理與驗證"""

        if "error" in generation_result:
            return generation_result

        generated_text = generation_result["generated_text"]

        # 1. 引用提取與驗證
        citations = await self._extract_and_validate_citations(
            generated_text, context_sources
        )

        # 2. 事實一致性檢查
        factual_consistency = await self._check_factual_consistency(
            generated_text, context_sources
        )

        # 3. 相關性評估
        relevance_score = await self._assess_response_relevance(
            generated_text, original_query
        )

        # 4. 安全性檢查
        safety_check = await self._perform_safety_check(generated_text)

        # 5. 品質綜合評分
        quality_score = self._calculate_generation_quality(
            citations, factual_consistency, relevance_score, safety_check
        )

        return {
            **generation_result,
            "citations": citations,
            "factual_consistency": factual_consistency,
            "relevance_score": relevance_score,
            "safety_check": safety_check,
            "quality_score": quality_score,
            "validation_status": "passed" if quality_score > 0.7 else "failed"
        }

class vLLMPerformanceMonitor:
    """vLLM 性能監控器"""

    def __init__(self):
        self.metrics_buffer = []
        self.performance_thresholds = {
            "max_latency_ms": 5000,
            "min_throughput_tokens_per_sec": 100,
            "max_gpu_memory_percent": 95,
            "max_error_rate_percent": 5
        }

    async def record_generation(self, **kwargs):
        """記錄生成性能指標"""

        metrics = {
            "timestamp": time.time(),
            **kwargs
        }

        self.metrics_buffer.append(metrics)

        # 保持緩衝區大小
        if len(self.metrics_buffer) > 1000:
            self.metrics_buffer.pop(0)

        # 檢查性能告警
        await self._check_performance_alerts(metrics)

    async def _check_performance_alerts(self, current_metrics: Dict):
        """檢查性能告警條件"""

        # 延遲告警
        if current_metrics.get("generation_time", 0) * 1000 > self.performance_thresholds["max_latency_ms"]:
            await self._trigger_alert("high_latency", current_metrics)

        # 計算最近的平均性能
        recent_metrics = self.metrics_buffer[-10:]  # 最近10次
        if len(recent_metrics) >= 5:
            avg_tokens_per_sec = sum(
                m.get("output_tokens", 0) / max(m.get("generation_time", 1), 0.001)
                for m in recent_metrics
            ) / len(recent_metrics)

            if avg_tokens_per_sec < self.performance_thresholds["min_throughput_tokens_per_sec"]:
                await self._trigger_alert("low_throughput", {
                    "current_throughput": avg_tokens_per_sec,
                    "threshold": self.performance_thresholds["min_throughput_tokens_per_sec"]
                })

    async def _trigger_alert(self, alert_type: str, metrics: Dict):
        """觸發性能告警"""

        alert_message = {
            "alert_type": alert_type,
            "timestamp": time.time(),
            "metrics": metrics,
            "severity": "warning" if alert_type == "low_throughput" else "critical"
        }

        # 實際實現中會發送到告警系統
        print(f"🚨 Performance Alert: {alert_message}")
```

---

## 3. 引用系統的理論與實現

### 3.1 自動引用生成理論

#### **來源歸屬的演算法框架**

**定義 3.1** (來源歸屬問題): 給定生成文本 $y = \{s_1, s_2, ..., s_n\}$ (句子序列) 和來源集合 $C = \{c_1, c_2, ..., c_m\}$，找到最優歸屬函數：

$$\text{Attribution}: \{s_1, ..., s_n\} \rightarrow \mathcal{P}(C)$$

使得歸屬準確度最大化：

$$\max \sum_{i=1}^{n} \text{Accuracy}(\text{Attribution}(s_i), \text{TrueSource}(s_i))$$

#### **語義相似度驗證算法**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from typing import Dict, List, Tuple
import re

class AutomaticCitationGenerator:
    """自動引用生成系統"""

    def __init__(self):
        # 語義相似度模型
        self.sentence_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )

        # 引用驗證模型
        self.citation_verifier = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )

        # 引用格式正則表達式
        self.citation_pattern = re.compile(r'\[SOURCE_(\d+)\]')

    async def generate_citations(self, generated_text: str,
                               source_documents: List[Dict]) -> Dict:
        """生成並驗證引用"""

        # 1. 句子分割
        sentences = await self._split_into_sentences(generated_text)

        # 2. 為每個句子找到最佳來源
        sentence_attributions = []
        for sentence in sentences:
            attribution = await self._find_best_source_attribution(
                sentence, source_documents
            )
            sentence_attributions.append(attribution)

        # 3. 生成引用增強文本
        citation_enhanced_text = await self._insert_citations(
            sentences, sentence_attributions
        )

        # 4. 驗證引用品質
        citation_quality = await self._validate_citation_quality(
            citation_enhanced_text, source_documents
        )

        # 5. 生成參考文獻
        bibliography = self._generate_bibliography(
            source_documents, sentence_attributions
        )

        return {
            "original_text": generated_text,
            "citation_enhanced_text": citation_enhanced_text,
            "sentence_attributions": sentence_attributions,
            "citation_quality": citation_quality,
            "bibliography": bibliography,
            "citation_coverage": len([a for a in sentence_attributions if a["has_citation"]]) / len(sentences)
        }

    async def _find_best_source_attribution(self, sentence: str,
                                          sources: List[Dict]) -> Dict:
        """為句子找到最佳來源歸屬"""

        if not sources:
            return {"has_citation": False, "reason": "no_sources"}

        # 計算語義相似度
        sentence_embedding = self.sentence_model.encode([sentence])[0]

        best_attribution = {
            "has_citation": False,
            "source_id": None,
            "confidence": 0.0,
            "similarity_score": 0.0,
            "evidence_text": ""
        }

        for i, source in enumerate(sources):
            source_content = source["content"]

            # 將來源分割為段落進行匹配
            source_paragraphs = self._split_into_paragraphs(source_content)

            for paragraph in source_paragraphs:
                # 計算語義相似度
                para_embedding = self.sentence_model.encode([paragraph])[0]
                similarity = self._cosine_similarity(sentence_embedding, para_embedding)

                # 使用交叉編碼器進行精確驗證
                if similarity > 0.5:  # 初步篩選
                    cross_encoder_score = self.citation_verifier.predict([
                        (sentence, paragraph)
                    ])[0]

                    # 綜合評分
                    combined_score = 0.6 * similarity + 0.4 * cross_encoder_score

                    if combined_score > best_attribution["confidence"]:
                        best_attribution = {
                            "has_citation": combined_score > 0.7,
                            "source_id": f"SOURCE_{i + 1}",
                            "confidence": combined_score,
                            "similarity_score": similarity,
                            "cross_encoder_score": cross_encoder_score,
                            "evidence_text": paragraph[:200] + "..." if len(paragraph) > 200 else paragraph
                        }

        return best_attribution

    async def _validate_citation_quality(self, cited_text: str,
                                       sources: List[Dict]) -> Dict:
        """驗證引用品質"""

        # 提取所有引用
        citations = self.citation_pattern.findall(cited_text)

        validation_results = {
            "total_citations": len(citations),
            "valid_citations": 0,
            "invalid_citations": [],
            "coverage_analysis": {},
            "accuracy_score": 0.0
        }

        for citation in set(citations):  # 去重
            source_idx = int(citation) - 1  # 轉換為索引

            if 0 <= source_idx < len(sources):
                validation_results["valid_citations"] += 1

                # 驗證引用準確性
                citation_accuracy = await self._verify_citation_accuracy(
                    cited_text, citation, sources[source_idx]
                )

                validation_results["coverage_analysis"][citation] = citation_accuracy
            else:
                validation_results["invalid_citations"].append({
                    "citation": citation,
                    "error": "source_not_found"
                })

        # 計算總體準確性
        if validation_results["valid_citations"] > 0:
            accuracy_scores = [
                acc["accuracy"] for acc in validation_results["coverage_analysis"].values()
            ]
            validation_results["accuracy_score"] = sum(accuracy_scores) / len(accuracy_scores)

        return validation_results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """計算餘弦相似度"""

        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    async def _verify_citation_accuracy(self, cited_text: str,
                                      citation: str,
                                      source: Dict) -> Dict:
        """驗證特定引用的準確性"""

        # 找到包含該引用的句子
        sentences_with_citation = []
        for sentence in cited_text.split('.'):
            if f"[SOURCE_{citation}]" in sentence:
                sentences_with_citation.append(sentence.strip())

        if not sentences_with_citation:
            return {"accuracy": 0.0, "reason": "citation_not_found_in_text"}

        # 驗證每個包含引用的句子
        accuracy_scores = []
        for sentence in sentences_with_citation:
            # 清除引用標記，只保留陳述內容
            clean_sentence = re.sub(r'\[SOURCE_\d+\]', '', sentence).strip()

            if len(clean_sentence) < 10:
                continue

            # 在來源中查找支持證據
            evidence_found = await self._find_supporting_evidence(
                clean_sentence, source["content"]
            )

            accuracy_scores.append(evidence_found["confidence"])

        if accuracy_scores:
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        else:
            avg_accuracy = 0.0

        return {
            "accuracy": avg_accuracy,
            "sentences_checked": len(sentences_with_citation),
            "evidence_quality": accuracy_scores
        }

    async def _find_supporting_evidence(self, statement: str,
                                      source_content: str) -> Dict:
        """在來源中查找支持證據"""

        # 將來源分割為可檢索的片段
        source_chunks = self._split_into_chunks(source_content, chunk_size=200)

        best_evidence = {"confidence": 0.0, "evidence_text": "", "chunk_index": -1}

        # 為每個片段計算支持度
        for i, chunk in enumerate(source_chunks):
            # 語義相似度
            similarity = await self._calculate_semantic_similarity(statement, chunk)

            # 詞彙重疊度
            lexical_overlap = self._calculate_lexical_overlap(statement, chunk)

            # 綜合置信度
            confidence = 0.7 * similarity + 0.3 * lexical_overlap

            if confidence > best_evidence["confidence"]:
                best_evidence = {
                    "confidence": confidence,
                    "evidence_text": chunk,
                    "chunk_index": i
                }

        return best_evidence

    def _calculate_lexical_overlap(self, text1: str, text2: str) -> float:
        """計算詞彙重疊度"""

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1:
            return 0.0

        intersection = words1 & words2
        return len(intersection) / len(words1)

    def _split_into_chunks(self, text: str, chunk_size: int = 200) -> List[str]:
        """將文本分割為片段"""

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _generate_bibliography(self, sources: List[Dict],
                             attributions: List[Dict]) -> List[Dict]:
        """生成參考文獻"""

        # 統計被引用的來源
        cited_sources = set()
        for attr in attributions:
            if attr.get("has_citation") and attr.get("source_id"):
                source_num = int(attr["source_id"].replace("SOURCE_", ""))
                cited_sources.add(source_num - 1)  # 轉為索引

        # 生成參考文獻條目
        bibliography = []
        for i in sorted(cited_sources):
            if i < len(sources):
                source = sources[i]
                bib_entry = {
                    "source_number": i + 1,
                    "title": source.get("metadata", {}).get("title", f"Document {i + 1}"),
                    "author": source.get("metadata", {}).get("author", "Unknown"),
                    "date": source.get("metadata", {}).get("date", "Unknown"),
                    "url": source.get("metadata", {}).get("url", ""),
                    "document_type": source.get("metadata", {}).get("type", "Document")
                }
                bibliography.append(bib_entry)

        return bibliography
```

---

## 4. 事實檢查與驗證系統

### 4.1 多層次事實驗證框架

#### **事實驗證的理論模型**

**定義 4.1** (事實陳述): 可以被客觀驗證為真或假的陳述。

**定理 4.1** (事實驗證的不完全性): 在開放域知識系統中，不存在完美的事實驗證算法，任何驗證系統都存在：

- **第一類錯誤** (假陽性): 將錯誤陳述判定為正確
- **第二類錯誤** (假陰性): 將正確陳述判定為錯誤

**優化目標**: 在給定的錯誤容忍度下，最大化驗證覆蓋率。

#### **多層次驗證架構**

```python
from transformers import pipeline
import spacy
from typing import Dict, List, Tuple, Optional

class MultiLevelFactChecker:
    """多層次事實檢查器"""

    def __init__(self):
        # NLI 模型用於蘊含關係檢查
        self.nli_model = pipeline(
            "text-classification",
            model="microsoft/deberta-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )

        # 事實性檢查模型
        self.factuality_checker = pipeline(
            "text-classification",
            model="tals/albert-xlarge-vitaminc-mnli",
            device=0 if torch.cuda.is_available() else -1
        )

        # NLP 處理工具
        self.nlp = spacy.load("en_core_web_lg")

    async def comprehensive_fact_check(self, generated_text: str,
                                     source_contexts: List[str],
                                     external_knowledge: Optional[Dict] = None) -> Dict:
        """全面事實檢查"""

        # 第一層: 陳述抽取
        factual_statements = await self._extract_factual_statements(generated_text)

        # 第二層: 上下文蘊含檢查
        entailment_results = await self._check_context_entailment(
            factual_statements, source_contexts
        )

        # 第三層: 外部知識驗證 (如果可用)
        external_validation = {}
        if external_knowledge:
            external_validation = await self._validate_against_external_kb(
                factual_statements, external_knowledge
            )

        # 第四層: 一致性檢查
        consistency_check = await self._check_internal_consistency(factual_statements)

        # 綜合評估
        overall_assessment = await self._synthesize_fact_check_results(
            entailment_results, external_validation, consistency_check
        )

        return {
            "factual_statements": factual_statements,
            "entailment_results": entailment_results,
            "external_validation": external_validation,
            "consistency_check": consistency_check,
            "overall_assessment": overall_assessment
        }

    async def _extract_factual_statements(self, text: str) -> List[Dict]:
        """提取事實陳述"""

        doc = self.nlp(text)
        factual_statements = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()

            if len(sentence_text) < 10:
                continue

            # 檢查是否為事實陳述 (vs 觀點、指令等)
            is_factual = await self._classify_statement_type(sentence_text)

            if is_factual["type"] == "factual":
                # 抽取關鍵實體和數值
                entities = [(ent.text, ent.label_) for ent in sent.ents]
                numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', sentence_text)
                dates = re.findall(r'\b\d{4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', sentence_text)

                factual_statements.append({
                    "text": sentence_text,
                    "sentence_id": len(factual_statements),
                    "entities": entities,
                    "numbers": numbers,
                    "dates": dates,
                    "factual_confidence": is_factual["confidence"],
                    "statement_type": "factual"
                })

        return factual_statements

    async def _check_context_entailment(self, statements: List[Dict],
                                      contexts: List[str]) -> Dict:
        """檢查上下文蘊含關係"""

        entailment_results = []
        combined_context = "\n\n".join(contexts)

        for statement in statements:
            statement_text = statement["text"]

            # 使用 NLI 模型檢查蘊含關係
            nli_result = self.nli_model(f"{combined_context} {statement_text}")

            # 解析 NLI 結果
            entailment_score = 0.0
            for result in nli_result:
                if result["label"] == "ENTAILMENT":
                    entailment_score = result["score"]
                    break

            # 使用事實性檢查模型進行二次驗證
            factuality_result = self.factuality_checker(f"{combined_context} [SEP] {statement_text}")

            factuality_score = 0.0
            for result in factuality_result:
                if result["label"] in ["SUPPORTS", "ENTAILMENT"]:
                    factuality_score = result["score"]
                    break

            # 綜合判斷
            combined_confidence = (entailment_score + factuality_score) / 2

            entailment_results.append({
                "statement": statement_text,
                "entailment_score": entailment_score,
                "factuality_score": factuality_score,
                "combined_confidence": combined_confidence,
                "supported": combined_confidence > 0.7,
                "evidence_strength": "strong" if combined_confidence > 0.8 else
                                   "moderate" if combined_confidence > 0.6 else "weak"
            })

        # 計算總體統計
        supported_count = sum(1 for r in entailment_results if r["supported"])
        total_statements = len(entailment_results)

        return {
            "statement_results": entailment_results,
            "overall_support_rate": supported_count / total_statements if total_statements > 0 else 0,
            "average_confidence": sum(r["combined_confidence"] for r in entailment_results) / total_statements if total_statements > 0 else 0,
            "unsupported_statements": [r for r in entailment_results if not r["supported"]]
        }

    async def _classify_statement_type(self, statement: str) -> Dict:
        """分類陳述類型"""

        # 事實陳述的語言特徵
        factual_indicators = [
            "is", "are", "was", "were", "has", "have", "will",
            "reports", "shows", "indicates", "found", "discovered"
        ]

        # 觀點陳述的語言特徵
        opinion_indicators = [
            "think", "believe", "feel", "opinion", "seems", "appears",
            "should", "must", "recommend", "suggest"
        ]

        statement_lower = statement.lower()

        factual_score = sum(1 for indicator in factual_indicators
                           if indicator in statement_lower)
        opinion_score = sum(1 for indicator in opinion_indicators
                          if indicator in statement_lower)

        if factual_score > opinion_score:
            return {"type": "factual", "confidence": 0.8}
        elif opinion_score > factual_score:
            return {"type": "opinion", "confidence": 0.8}
        else:
            return {"type": "uncertain", "confidence": 0.5}
```

---

## 5. 生產級部署最佳實踐

### 5.1 高可用性架構設計

#### **負載平衡與故障切換**

```python
import asyncio
from typing import Dict, List, Optional, Any
import random
import time

class LLMLoadBalancer:
    """LLM 負載平衡器"""

    def __init__(self, model_instances: Dict[str, Dict]):
        self.instances = model_instances
        self.health_checker = InstanceHealthChecker()
        self.request_router = RequestRouter()

        # 負載平衡策略
        self.balancing_strategies = {
            "round_robin": self._round_robin_selection,
            "least_connections": self._least_connections_selection,
            "weighted_response_time": self._weighted_response_time_selection,
            "resource_aware": self._resource_aware_selection
        }

        self.current_strategy = "resource_aware"

    async def route_generation_request(self, request: GenerationRequest) -> Dict:
        """路由生成請求到最佳實例"""

        # 1. 健康實例篩選
        healthy_instances = await self._get_healthy_instances()

        if not healthy_instances:
            return {
                "error": "No healthy instances available",
                "status": "service_unavailable"
            }

        # 2. 請求路由
        selected_instance = await self._select_optimal_instance(
            healthy_instances, request
        )

        # 3. 執行請求
        try:
            result = await self._execute_on_instance(selected_instance, request)

            # 4. 記錄性能指標
            await self._update_instance_metrics(selected_instance, result, success=True)

            return result

        except Exception as e:
            # 故障切換
            await self._handle_instance_failure(selected_instance, str(e))

            # 嘗試備用實例
            backup_result = await self._try_backup_instances(
                healthy_instances, request, exclude=[selected_instance]
            )

            return backup_result

    async def _get_healthy_instances(self) -> List[str]:
        """獲取健康的實例列表"""

        healthy_instances = []

        for instance_id, instance_config in self.instances.items():
            health_status = await self.health_checker.check_instance_health(
                instance_id, instance_config
            )

            if health_status["status"] == "healthy":
                healthy_instances.append(instance_id)

        return healthy_instances

    async def _select_optimal_instance(self, healthy_instances: List[str],
                                     request: GenerationRequest) -> str:
        """選擇最佳實例"""

        strategy_func = self.balancing_strategies[self.current_strategy]
        return await strategy_func(healthy_instances, request)

    async def _resource_aware_selection(self, instances: List[str],
                                      request: GenerationRequest) -> str:
        """資源感知的實例選擇"""

        instance_scores = {}

        for instance_id in instances:
            instance_config = self.instances[instance_id]

            # 獲取當前資源使用情況
            resource_usage = await self._get_instance_resource_usage(instance_id)

            # 計算負載分數 (越低越好)
            load_score = (
                0.4 * resource_usage.get("gpu_utilization", 0) +
                0.3 * resource_usage.get("memory_utilization", 0) +
                0.2 * resource_usage.get("cpu_utilization", 0) +
                0.1 * resource_usage.get("queue_depth", 0) / 100  # 標準化
            )

            # 考慮實例性能權重
            performance_weight = instance_config.get("performance_weight", 1.0)

            # 綜合評分 (越低越好)
            instance_scores[instance_id] = load_score / performance_weight

        # 選擇負載最低的實例
        best_instance = min(instance_scores.keys(), key=lambda k: instance_scores[k])

        return best_instance

    async def _execute_on_instance(self, instance_id: str,
                                 request: GenerationRequest) -> Dict:
        """在指定實例上執行請求"""

        instance_config = self.instances[instance_id]
        instance_client = instance_config["client"]

        start_time = time.time()

        # 執行生成請求
        result = await instance_client.generate(
            prompt=request.prompt,
            sampling_params=request.sampling_params
        )

        execution_time = time.time() - start_time

        return {
            "result": result,
            "instance_id": instance_id,
            "execution_time": execution_time,
            "status": "success"
        }

    async def _handle_instance_failure(self, instance_id: str, error: str):
        """處理實例故障"""

        # 標記實例為不健康
        await self.health_checker.mark_instance_unhealthy(instance_id, error)

        # 記錄故障
        await self._log_instance_failure(instance_id, error)

        # 觸發告警
        await self._trigger_failure_alert(instance_id, error)

    async def _try_backup_instances(self, available_instances: List[str],
                                  request: GenerationRequest,
                                  exclude: List[str]) -> Dict:
        """嘗試備用實例"""

        backup_instances = [i for i in available_instances if i not in exclude]

        if not backup_instances:
            return {
                "error": "No backup instances available",
                "status": "all_instances_failed"
            }

        # 選擇備用實例
        backup_instance = backup_instances[0]  # 簡化選擇

        try:
            return await self._execute_on_instance(backup_instance, request)
        except Exception as e:
            return {
                "error": f"Backup instance also failed: {str(e)}",
                "status": "backup_failed"
            }
```

---

## 6. 本章總結

### 6.1 核心學習要點

1. **理論基礎**: 深度理解 Transformer 架構和生成控制理論
2. **系統設計**: 掌握企業級 LLM 服務的架構設計原則
3. **品質保證**: 建立完整的事實檢查和引用驗證機制
4. **性能優化**: 實現高效能、高可用的生產部署策略

### 6.2 實踐指導原則

1. **品質優於速度**: 在企業環境中，準確性比生成速度更重要
2. **可追溯性**: 所有生成內容都應有明確的來源歸屬
3. **漸進式部署**: 從低風險場景開始，逐步擴展到關鍵業務
4. **持續監控**: 建立完善的性能和品質監控機制

### 6.3 下章預告

第5章將深入探討 RAG 系統的評估理論與監控體系，重點分析如何科學地測量和持續改進系統性能，這是確保企業級 RAG 系統長期成功的關鍵。

---

**課程評估**: 本章內容在期末考試中占20%權重，重點考查 LLM 服務架構和品質控制能力。

**項目要求**: 學生需實現一個完整的 LLM 服務系統，包括負載平衡、引用生成和事實檢查功能。