# RAG 評估理論與指標體系
## 大學教科書 第5章：檢索增強生成系統的科學評估

**課程編號**: CS785 - 企業級檢索增強生成系統
**章節**: 第5章 評估與監控
**學習時數**: 6小時
**先修課程**: 統計學基礎, 機器學習評估, 第0-4章
**作者**: ML評估研究團隊 & RAGAS開發團隊合作
**最後更新**: 2025-01-06

---

## 📚 學習目標 (Learning Objectives)

完成本章學習後，學生應能夠:

1. **評估理論**: 掌握 RAG 系統評估的理論框架和數學基礎
2. **指標體系**: 理解並應用 RAGAS 評估框架的核心指標
3. **實驗設計**: 設計科學的 RAG 系統性能評估實驗
4. **監控系統**: 建立生產環境的持續監控和品質保證機制

---

## 1. RAG 評估的理論框架

### 1.1 評估複雜性的根源分析

#### **多維度評估挑戰**

RAG 系統的評估複雜性源於其**多階段流水線特性**，每個階段都需要獨特的評估方法：

**定理 1.1** (RAG 評估的不可分解性): RAG 系統的整體性能不等於各組件性能的簡單加權和，存在顯著的**交互效應**：

$$\text{Performance}_{RAG} \neq \sum_{i} w_i \cdot \text{Performance}_i$$

而是：
$$\text{Performance}_{RAG} = f(\text{Retrieval}, \text{Augmentation}, \text{Generation}) + \sum_{i<j} \text{Interaction}_{ij}$$

其中 $\text{Interaction}_{ij}$ 表示組件間的交互效應。

**證明思路**: 檢索錯誤可能被生成模型的先驗知識補償，而檢索噪音可能被上下文工程技術消除，這些交互效應使得分解評估不足以預測整體性能。□

#### **評估維度的數學建模**

基於 Es et al. (2023)[^18] 的 RAGAS 框架，RAG 系統評估包含以下核心維度：

**維度 1.1** (忠實度 Faithfulness): 生成答案與檢索上下文的事實一致性

$$\text{Faithfulness} = \frac{|\text{支持的陳述}|}{|\text{總陳述}|}$$

**維度 1.2** (答案相關性 Answer Relevancy): 生成答案對原始問題的相關程度

$$\text{Answer Relevancy} = \frac{1}{|Q|} \sum_{q_i \in Q} \text{Similarity}(q, q_i)$$

其中 $Q = \{q_1, q_2, ..., q_n\}$ 是基於答案生成的問題集合。

**維度 1.3** (上下文精確度 Context Precision): 檢索上下文中相關信息的比例

$$\text{Context Precision} = \frac{|\text{相關上下文}|}{|\text{總檢索上下文}|}$$

**維度 1.4** (上下文召回率 Context Recall): 回答問題所需信息在檢索上下文中的覆蓋率

$$\text{Context Recall} = \frac{|\text{檢索到的必需信息}|}{|\text{回答所需的總信息}|}$$

### 1.2 評估指標的信息論分析

#### **信息熵視角的評估**

**定義 1.1** (評估信息熵): RAG 系統評估的信息熵定義為：

$$H_{eval} = -\sum_{m \in M} P(m) \log P(m)$$

其中 $M$ 為評估指標集合，$P(m)$ 為指標 $m$ 的重要性權重。

**推論 1.1** (最大熵評估原理): 在沒有先驗知識的情況下，應選擇使評估信息熵最大的指標組合，以獲得最全面的性能評估。

---

## 2. RAGAS 評估框架深度解析

### 2.1 忠實度 (Faithfulness) 的計算理論

#### **陳述分解與事實驗證**

**算法 2.1** (基於 LLM 的陳述分解):

```python
from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass

@dataclass
class Statement:
    """事實陳述數據結構"""
    text: str
    statement_id: str
    confidence: float
    source_span: Optional[Tuple[int, int]]  # 在原文中的位置

class FaithfulnessEvaluator:
    """忠實度評估器"""

    def __init__(self, llm_evaluator: Any):
        self.llm_evaluator = llm_evaluator

    async def calculate_faithfulness(self, answer: str,
                                   contexts: List[str]) -> Dict[str, Any]:
        """
        計算忠實度分數

        基於 Es et al. (2023) RAGAS 框架實現
        """

        # 步驟1: 分解答案為原子陳述
        statements = await self._decompose_into_statements(answer)

        if not statements:
            return {"faithfulness": 0.0, "details": "No statements found"}

        # 步驟2: 驗證每個陳述
        verification_results = []
        for statement in statements:
            verification = await self._verify_statement(statement, contexts)
            verification_results.append(verification)

        # 步驟3: 計算忠實度分數
        supported_count = sum(1 for v in verification_results if v["supported"])
        faithfulness_score = supported_count / len(statements)

        return {
            "faithfulness": faithfulness_score,
            "total_statements": len(statements),
            "supported_statements": supported_count,
            "statement_details": verification_results
        }

    async def _decompose_into_statements(self, answer: str) -> List[Statement]:
        """將答案分解為原子陳述"""

        prompt = f"""
        請將以下答案分解為獨立的事實陳述，每個陳述應該是一個可以獨立驗證的原子事實。

        答案: {answer}

        請以以下格式返回：
        1. [陳述1]
        2. [陳述2]
        ...

        分解結果:
        """

        response = await self.llm_evaluator.generate(prompt, temperature=0.1)

        # 解析陳述
        statements = []
        lines = response.strip().split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if line and re.match(r'^\d+\.', line):
                statement_text = re.sub(r'^\d+\.\s*', '', line)
                if statement_text:
                    statement = Statement(
                        text=statement_text,
                        statement_id=f"stmt_{i}",
                        confidence=1.0,  # 初始置信度
                        source_span=None
                    )
                    statements.append(statement)

        return statements

    async def _verify_statement(self, statement: Statement,
                               contexts: List[str]) -> Dict[str, Any]:
        """驗證陳述是否被上下文支持"""

        # 將所有上下文合併
        combined_context = "\n\n".join(contexts)

        # 構建驗證提示
        prompt = f"""
        請判斷以下陳述是否被給定的上下文支持。

        陳述: {statement.text}

        上下文:
        {combined_context}

        請回答以下問題：
        1. 該陳述是否被上下文明確支持？ (是/否)
        2. 支持該陳述的具體證據是什麼？
        3. 支持的信心程度如何？ (0-1分)

        請以JSON格式回答：
        {{
            "supported": true/false,
            "evidence": "支持證據的文本",
            "confidence": 0.95
        }}
        """

        response = await self.llm_evaluator.generate(prompt, temperature=0.0)

        try:
            verification_result = self._parse_json_response(response)
            verification_result["statement_text"] = statement.text
            return verification_result
        except Exception as e:
            return {
                "supported": False,
                "evidence": "",
                "confidence": 0.0,
                "error": str(e),
                "statement_text": statement.text
            }

    def _parse_json_response(self, response: str) -> Dict:
        """解析 JSON 格式的回應"""
        import json

        # 嘗試提取 JSON 部分
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            json_text = json_match.group()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

        # 如果 JSON 解析失敗，使用規則提取
        supported = "true" in response.lower() or "是" in response
        confidence_match = re.search(r'(\d+\.?\d*)', response)
        confidence = float(confidence_match.group()) if confidence_match else 0.5

        return {
            "supported": supported,
            "evidence": response[:200],
            "confidence": confidence
        }
```

### 2.2 答案相關性 (Answer Relevancy) 的理論模型

#### **反向問題生成方法**

**原理**: 基於生成的答案，使用 LLM 反向生成可能的問題，通過這些問題與原始問題的相似度來評估答案相關性。

**數學模型**: 設原始問題為 $q$，答案為 $a$，反向生成的問題集合為 $Q' = \{q'_1, q'_2, ..., q'_n\}$，則答案相關性為：

$$\text{Answer Relevancy} = \frac{1}{n} \sum_{i=1}^{n} \text{Similarity}(q, q'_i)$$

**算法 2.2** (答案相關性評估):

```python
class AnswerRelevancyEvaluator:
    """答案相關性評估器"""

    def __init__(self, llm_evaluator: Any, embedding_model: Any):
        self.llm_evaluator = llm_evaluator
        self.embedding_model = embedding_model

    async def calculate_answer_relevancy(self, question: str,
                                       answer: str,
                                       num_questions: int = 3) -> Dict[str, Any]:
        """
        計算答案相關性分數

        基於 Es et al. (2023) 的反向問題生成方法
        """

        # 步驟1: 基於答案生成問題
        generated_questions = await self._generate_questions_from_answer(
            answer, num_questions
        )

        if not generated_questions:
            return {"answer_relevancy": 0.0, "details": "No questions generated"}

        # 步驟2: 計算問題相似度
        original_embedding = self.embedding_model.encode([question])[0]
        generated_embeddings = self.embedding_model.encode(generated_questions)

        # 步驟3: 計算平均相似度
        similarities = []
        for gen_embedding in generated_embeddings:
            similarity = cosine_similarity(
                [original_embedding], [gen_embedding]
            )[0][0]
            similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities)

        return {
            "answer_relevancy": avg_similarity,
            "generated_questions": generated_questions,
            "individual_similarities": similarities,
            "question_quality": await self._assess_question_quality(generated_questions)
        }

    async def _generate_questions_from_answer(self, answer: str,
                                            num_questions: int) -> List[str]:
        """基於答案生成問題"""

        prompt = f"""
        基於以下答案，生成 {num_questions} 個可能導致這個答案的問題。
        問題應該：
        1. 邏輯合理且自然
        2. 涵蓋答案的主要信息點
        3. 具有不同的詢問角度

        答案: {answer}

        請生成問題：
        """

        response = await self.llm_evaluator.generate(prompt, temperature=0.3)

        # 解析生成的問題
        questions = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line and ('?' in line or '？' in line):
                # 清理問題格式
                question = re.sub(r'^\d+\.?\s*', '', line).strip()
                if len(question) > 10:  # 過濾過短問題
                    questions.append(question)

        return questions[:num_questions]

    async def _assess_question_quality(self, questions: List[str]) -> Dict[str, float]:
        """評估生成問題的品質"""

        if not questions:
            return {"diversity": 0.0, "naturalness": 0.0, "complexity": 0.0}

        # 1. 多樣性評估
        question_embeddings = self.embedding_model.encode(questions)
        diversity_score = await self._calculate_diversity(question_embeddings)

        # 2. 自然度評估
        naturalness_scores = []
        for question in questions:
            naturalness = await self._assess_question_naturalness(question)
            naturalness_scores.append(naturalness)

        avg_naturalness = sum(naturalness_scores) / len(naturalness_scores)

        # 3. 複雜度評估
        complexity_scores = []
        for question in questions:
            complexity = await self._assess_question_complexity(question)
            complexity_scores.append(complexity)

        avg_complexity = sum(complexity_scores) / len(complexity_scores)

        return {
            "diversity": diversity_score,
            "naturalness": avg_naturalness,
            "complexity": avg_complexity
        }

    async def _calculate_diversity(self, embeddings: np.ndarray) -> float:
        """計算問題集合的多樣性"""

        if len(embeddings) <= 1:
            return 1.0

        # 計算兩兩相似度
        similarity_matrix = cosine_similarity(embeddings)

        # 去除對角線元素 (自相似度)
        np.fill_diagonal(similarity_matrix, 0)

        # 多樣性 = 1 - 平均相似度
        avg_similarity = np.mean(similarity_matrix)
        diversity = 1.0 - avg_similarity

        return max(0.0, diversity)
```

### 2.3 上下文精確度與召回率

#### **信息檢索理論的延伸**

**定義 2.1** (上下文級精確度): 在檢索上下文中，與回答問題相關的信息比例：

$$\text{Context Precision@k} = \frac{1}{k} \sum_{i=1}^{k} \text{Relevance}(c_i, q)$$

其中 $c_i$ 為第 $i$ 個檢索到的上下文片段。

**定義 2.2** (上下文級召回率): 回答問題所需的信息在檢索上下文中的覆蓋程度：

$$\text{Context Recall} = \frac{|\text{檢索到的必需信息} \cap \text{標準答案信息}|}{|\text{標準答案信息}|}$$

#### **實現算法**

```python
class ContextEvaluator:
    """上下文品質評估器"""

    def __init__(self, llm_evaluator: Any):
        self.llm_evaluator = llm_evaluator

    async def calculate_context_precision(self, question: str,
                                        contexts: List[str]) -> Dict[str, Any]:
        """計算上下文精確度"""

        if not contexts:
            return {"context_precision": 0.0}

        relevance_scores = []
        detailed_assessments = []

        for i, context in enumerate(contexts):
            # 評估每個上下文的相關性
            relevance = await self._assess_context_relevance(question, context)
            relevance_scores.append(relevance["score"])

            detailed_assessments.append({
                "context_index": i,
                "context_preview": context[:100] + "..." if len(context) > 100 else context,
                "relevance_score": relevance["score"],
                "relevance_reasoning": relevance.get("reasoning", "")
            })

        # 計算精確度
        precision = sum(relevance_scores) / len(relevance_scores)

        return {
            "context_precision": precision,
            "individual_scores": relevance_scores,
            "detailed_assessments": detailed_assessments,
            "num_contexts": len(contexts)
        }

    async def calculate_context_recall(self, question: str,
                                     contexts: List[str],
                                     ground_truth_answer: str) -> Dict[str, Any]:
        """計算上下文召回率"""

        if not contexts or not ground_truth_answer:
            return {"context_recall": 0.0}

        # 步驟1: 從標準答案中提取關鍵信息
        required_info = await self._extract_required_information(
            question, ground_truth_answer
        )

        if not required_info:
            return {"context_recall": 0.0, "details": "No required information identified"}

        # 步驟2: 檢查每個關鍵信息是否在上下文中
        coverage_results = []
        for info_item in required_info:
            coverage = await self._check_information_coverage(
                info_item, contexts
            )
            coverage_results.append(coverage)

        # 步驟3: 計算召回率
        covered_count = sum(1 for c in coverage_results if c["covered"])
        recall = covered_count / len(required_info)

        return {
            "context_recall": recall,
            "required_information": required_info,
            "coverage_results": coverage_results,
            "covered_items": covered_count,
            "total_required": len(required_info)
        }

    async def _assess_context_relevance(self, question: str,
                                      context: str) -> Dict[str, Any]:
        """評估上下文與問題的相關性"""

        prompt = f"""
        評估以下上下文對於回答問題的相關性。

        問題: {question}

        上下文: {context}

        請評估：
        1. 該上下文是否包含與問題相關的信息？
        2. 相關性程度如何？(0-1分，1表示高度相關)
        3. 具體哪部分內容是相關的？

        請以JSON格式回答：
        {{
            "relevant": true/false,
            "score": 0.85,
            "reasoning": "相關性分析",
            "relevant_parts": "相關內容摘要"
        }}
        """

        response = await self.llm_evaluator.generate(prompt, temperature=0.1)

        try:
            result = self._parse_json_response(response)
            result["score"] = float(result.get("score", 0.0))
            return result
        except Exception:
            # 備用解析邏輯
            relevant = "relevant" in response.lower() or "相關" in response
            return {
                "relevant": relevant,
                "score": 0.7 if relevant else 0.2,
                "reasoning": response[:200],
                "relevant_parts": ""
            }

    async def _extract_required_information(self, question: str,
                                          ground_truth: str) -> List[str]:
        """從標準答案中提取回答問題所需的關鍵信息"""

        prompt = f"""
        分析標準答案，提取回答以下問題所必需的關鍵信息點。

        問題: {question}
        標準答案: {ground_truth}

        請列出回答該問題必須包含的關鍵信息點：
        1. [信息點1]
        2. [信息點2]
        ...

        關鍵信息點:
        """

        response = await self.llm_evaluator.generate(prompt, temperature=0.1)

        # 解析信息點
        info_items = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line and re.match(r'^\d+\.', line):
                info_text = re.sub(r'^\d+\.\s*', '', line).strip()
                if info_text and len(info_text) > 5:
                    info_items.append(info_text)

        return info_items

    async def _check_information_coverage(self, required_info: str,
                                        contexts: List[str]) -> Dict[str, Any]:
        """檢查必需信息是否被上下文覆蓋"""

        combined_contexts = "\n\n".join(contexts)

        prompt = f"""
        檢查以下必需信息是否在給定的上下文中被覆蓋。

        必需信息: {required_info}

        上下文:
        {combined_contexts}

        請判斷：
        1. 該信息是否在上下文中出現？
        2. 覆蓋程度如何？(0-1分)
        3. 在哪個部分找到該信息？

        請以JSON格式回答：
        {{
            "covered": true/false,
            "coverage_score": 0.9,
            "location": "信息在上下文中的位置描述"
        }}
        """

        response = await self.llm_evaluator.generate(prompt, temperature=0.1)

        try:
            result = self._parse_json_response(response)
            result["required_info"] = required_info
            return result
        except Exception:
            # 簡化的覆蓋檢查
            covered = any(self._text_overlap(required_info, context) > 0.3
                         for context in contexts)
            return {
                "covered": covered,
                "coverage_score": 0.7 if covered else 0.1,
                "location": "automatic_detection",
                "required_info": required_info
            }

    def _text_overlap(self, text1: str, text2: str) -> float:
        """計算兩個文本的重疊度"""

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1:
            return 0.0

        overlap = len(words1.intersection(words2))
        return overlap / len(words1)
```

---

## 3. 企業級評估體系設計

### 3.1 分層評估架構

#### **三層評估模型**

**層級 3.1** (RAG 評估金字塔):

```
                    ┌─────────────────┐
                    │   業務層評估     │  ← 用戶滿意度、業務KPI
                    │  (Business)     │
                    └─────────────────┘
                            ▲
                    ┌─────────────────┐
                    │   系統層評估     │  ← Faithfulness、Relevancy
                    │  (System)       │
                    └─────────────────┘
                            ▲
                    ┌─────────────────┐
                    │   組件層評估     │  ← Retrieval、Generation
                    │ (Component)     │
                    └─────────────────┘
```

**實現架構**:

```python
class EnterpriseRAGEvaluationFramework:
    """企業級 RAG 評估框架"""

    def __init__(self):
        # 組件層評估器
        self.component_evaluators = {
            "retrieval": RetrievalEvaluator(),
            "reranking": RerankingEvaluator(),
            "generation": GenerationEvaluator()
        }

        # 系統層評估器
        self.system_evaluators = {
            "faithfulness": FaithfulnessEvaluator(),
            "relevancy": AnswerRelevancyEvaluator(),
            "context_precision": ContextEvaluator(),
            "context_recall": ContextEvaluator()
        }

        # 業務層評估器
        self.business_evaluators = {
            "user_satisfaction": UserSatisfactionEvaluator(),
            "task_success": TaskSuccessEvaluator(),
            "cost_effectiveness": CostEffectivenessEvaluator()
        }

    async def comprehensive_evaluation(self, test_dataset: List[Dict],
                                     rag_system: Any) -> Dict[str, Any]:
        """執行全面評估"""

        results = {
            "component_level": {},
            "system_level": {},
            "business_level": {},
            "overall_assessment": {}
        }

        # 組件層評估
        for component_name, evaluator in self.component_evaluators.items():
            print(f"評估組件: {component_name}")
            component_result = await evaluator.evaluate(test_dataset, rag_system)
            results["component_level"][component_name] = component_result

        # 系統層評估
        for metric_name, evaluator in self.system_evaluators.items():
            print(f"評估指標: {metric_name}")
            metric_result = await evaluator.evaluate(test_dataset, rag_system)
            results["system_level"][metric_name] = metric_result

        # 業務層評估
        for business_metric, evaluator in self.business_evaluators.items():
            print(f"業務評估: {business_metric}")
            business_result = await evaluator.evaluate(test_dataset, rag_system)
            results["business_level"][business_metric] = business_result

        # 綜合評估
        overall_assessment = await self._calculate_overall_assessment(results)
        results["overall_assessment"] = overall_assessment

        return results

    async def _calculate_overall_assessment(self, evaluation_results: Dict) -> Dict:
        """計算綜合評估分數"""

        # 權重配置
        weights = {
            "component_level": 0.2,
            "system_level": 0.5,
            "business_level": 0.3
        }

        weighted_scores = {}

        # 組件層綜合分數
        component_scores = evaluation_results["component_level"]
        component_avg = sum(score.get("overall_score", 0.0)
                           for score in component_scores.values()) / len(component_scores)
        weighted_scores["component"] = component_avg * weights["component_level"]

        # 系統層綜合分數
        system_scores = evaluation_results["system_level"]
        system_avg = sum(score.get("score", 0.0)
                        for score in system_scores.values()) / len(system_scores)
        weighted_scores["system"] = system_avg * weights["system_level"]

        # 業務層綜合分數
        business_scores = evaluation_results["business_level"]
        business_avg = sum(score.get("score", 0.0)
                          for score in business_scores.values()) / len(business_scores)
        weighted_scores["business"] = business_avg * weights["business_level"]

        # 總體分數
        overall_score = sum(weighted_scores.values())

        return {
            "overall_score": overall_score,
            "weighted_scores": weighted_scores,
            "grade": self._assign_performance_grade(overall_score),
            "strengths": self._identify_strengths(evaluation_results),
            "weaknesses": self._identify_weaknesses(evaluation_results),
            "improvement_recommendations": self._generate_recommendations(evaluation_results)
        }

    def _assign_performance_grade(self, score: float) -> str:
        """分配性能等級"""
        if score >= 0.9:
            return "A+ (優秀)"
        elif score >= 0.8:
            return "A (良好)"
        elif score >= 0.7:
            return "B (合格)"
        elif score >= 0.6:
            return "C (需改進)"
        else:
            return "D (不合格)"

    def _identify_strengths(self, results: Dict) -> List[str]:
        """識別系統優勢"""

        strengths = []

        # 檢查各層級的高分項目
        for level_name, level_results in results.items():
            if level_name == "overall_assessment":
                continue

            for metric, result in level_results.items():
                score = result.get("score", result.get("overall_score", 0.0))
                if score > 0.8:
                    strengths.append(f"{level_name}.{metric}: {score:.2f}")

        return strengths

    def _identify_weaknesses(self, results: Dict) -> List[str]:
        """識別系統弱點"""

        weaknesses = []

        for level_name, level_results in results.items():
            if level_name == "overall_assessment":
                continue

            for metric, result in level_results.items():
                score = result.get("score", result.get("overall_score", 0.0))
                if score < 0.6:
                    weaknesses.append(f"{level_name}.{metric}: {score:.2f}")

        return weaknesses
```

---

## 4. 線上監控與品質保證

### 4.1 實時評估系統設計

#### **流式評估架構**

**系統 4.1** (實時 RAG 品質監控):

```python
import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class RealTimeQualityMonitor:
    """實時品質監控系統"""

    def __init__(self, evaluation_config: Dict):
        self.config = evaluation_config
        self.metrics_buffer = deque(maxlen=1000)  # 滾動窗口
        self.alert_thresholds = self._load_alert_thresholds()
        self.quality_trends = QualityTrendAnalyzer()

    async def monitor_query_execution(self, query: str, response: Dict,
                                    user_feedback: Optional[Dict] = None) -> Dict:
        """監控查詢執行的品質指標"""

        # 實時品質評估
        quality_metrics = await self._quick_quality_assessment(query, response)

        # 添加用戶反饋 (如果有)
        if user_feedback:
            quality_metrics["user_satisfaction"] = user_feedback.get("rating", 0.0)
            quality_metrics["user_helpful"] = user_feedback.get("helpful", False)

        # 記錄到緩衝區
        timestamp = datetime.now()
        self.metrics_buffer.append({
            "timestamp": timestamp,
            "query": query,
            "metrics": quality_metrics,
            "response_metadata": response.get("metadata", {})
        })

        # 檢查告警條件
        alerts = await self._check_alert_conditions(quality_metrics, timestamp)

        # 更新趨勢分析
        await self.quality_trends.update_trends(quality_metrics, timestamp)

        return {
            "quality_metrics": quality_metrics,
            "alerts": alerts,
            "monitoring_status": "active"
        }

    async def _quick_quality_assessment(self, query: str,
                                      response: Dict) -> Dict[str, float]:
        """快速品質評估 (適用於實時監控)"""

        metrics = {}

        # 1. 響應時間指標
        processing_time = response.get("processing_time_ms", 0)
        latency_score = self._calculate_latency_score(processing_time)
        metrics["latency_score"] = latency_score

        # 2. 來源品質指標
        sources = response.get("sources", [])
        source_quality = await self._assess_source_quality(sources)
        metrics["source_quality"] = source_quality

        # 3. 回答完整性 (簡化版本)
        answer = response.get("answer", "")
        answer_completeness = await self._estimate_answer_completeness(query, answer)
        metrics["answer_completeness"] = answer_completeness

        # 4. 引用覆蓋率
        citation_coverage = self._calculate_citation_coverage(answer, sources)
        metrics["citation_coverage"] = citation_coverage

        return metrics

    def _calculate_latency_score(self, processing_time_ms: float) -> float:
        """計算延遲分數 (越低越好)"""

        # SLO 目標: p95 < 500ms
        if processing_time_ms <= 200:
            return 1.0
        elif processing_time_ms <= 500:
            return 1.0 - (processing_time_ms - 200) / 300 * 0.3
        elif processing_time_ms <= 1000:
            return 0.7 - (processing_time_ms - 500) / 500 * 0.4
        else:
            return max(0.0, 0.3 - (processing_time_ms - 1000) / 2000 * 0.3)

    async def _assess_source_quality(self, sources: List[Dict]) -> float:
        """評估檢索來源的品質"""

        if not sources:
            return 0.0

        quality_scores = []

        for source in sources:
            score = 0.0

            # 來源可信度
            if source.get("confidence", 0) > 0.8:
                score += 0.3

            # 內容長度合理性
            content_length = len(source.get("content", ""))
            if 50 <= content_length <= 2000:
                score += 0.2
            elif content_length > 2000:
                score += 0.1

            # 元數據完整性
            metadata = source.get("metadata", {})
            if metadata.get("title") and metadata.get("timestamp"):
                score += 0.3

            # 相關性分數
            relevance = source.get("score", 0.0)
            score += 0.2 * min(1.0, relevance)

            quality_scores.append(score)

        return sum(quality_scores) / len(quality_scores)

    async def _check_alert_conditions(self, metrics: Dict[str, float],
                                    timestamp: datetime) -> List[Dict]:
        """檢查告警條件"""

        alerts = []

        # 檢查各項指標
        for metric_name, value in metrics.items():
            if metric_name in self.alert_thresholds:
                threshold = self.alert_thresholds[metric_name]

                if (threshold.get("type") == "min" and
                    value < threshold["value"]):
                    alerts.append({
                        "type": "quality_degradation",
                        "metric": metric_name,
                        "current_value": value,
                        "threshold": threshold["value"],
                        "severity": threshold.get("severity", "warning"),
                        "timestamp": timestamp
                    })

                elif (threshold.get("type") == "max" and
                      value > threshold["value"]):
                    alerts.append({
                        "type": "performance_degradation",
                        "metric": metric_name,
                        "current_value": value,
                        "threshold": threshold["value"],
                        "severity": threshold.get("severity", "warning"),
                        "timestamp": timestamp
                    })

        # 檢查趨勢告警
        trend_alerts = await self.quality_trends.check_trend_alerts(timestamp)
        alerts.extend(trend_alerts)

        return alerts

    def _load_alert_thresholds(self) -> Dict[str, Dict]:
        """載入告警閾值配置"""

        return {
            "faithfulness": {
                "type": "min",
                "value": 0.8,
                "severity": "warning"
            },
            "answer_relevancy": {
                "type": "min",
                "value": 0.7,
                "severity": "warning"
            },
            "context_precision": {
                "type": "min",
                "value": 0.6,
                "severity": "info"
            },
            "latency_score": {
                "type": "min",
                "value": 0.7,
                "severity": "critical"
            },
            "source_quality": {
                "type": "min",
                "value": 0.5,
                "severity": "warning"
            }
        }

class QualityTrendAnalyzer:
    """品質趨勢分析器"""

    def __init__(self):
        self.trend_window = timedelta(hours=24)  # 24小時趨勢窗口
        self.metrics_history = {}

    async def update_trends(self, metrics: Dict[str, float],
                           timestamp: datetime):
        """更新品質趨勢"""

        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = deque(maxlen=1000)

            self.metrics_history[metric_name].append({
                "timestamp": timestamp,
                "value": value
            })

    async def check_trend_alerts(self, current_time: datetime) -> List[Dict]:
        """檢查趨勢告警"""

        alerts = []

        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:  # 需要足夠的歷史數據
                continue

            # 計算最近趨勢
            recent_data = [
                record for record in history
                if current_time - record["timestamp"] <= self.trend_window
            ]

            if len(recent_data) < 5:
                continue

            # 計算趨勢斜率
            timestamps = [(r["timestamp"] - current_time).total_seconds()
                         for r in recent_data]
            values = [r["value"] for r in recent_data]

            trend_slope = self._calculate_trend_slope(timestamps, values)

            # 檢查下降趨勢
            if trend_slope < -0.1:  # 顯著下降趨勢
                alerts.append({
                    "type": "negative_trend",
                    "metric": metric_name,
                    "trend_slope": trend_slope,
                    "severity": "warning",
                    "timestamp": current_time,
                    "description": f"{metric_name} 呈現下降趨勢 (斜率: {trend_slope:.3f})"
                })

        return alerts

    def _calculate_trend_slope(self, x: List[float], y: List[float]) -> float:
        """計算線性趨勢斜率"""

        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        # 線性回歸斜率
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        return slope
```

---

## 5. A/B 測試與持續改進

### 5.1 RAG 系統 A/B 測試設計

#### **實驗設計原理**

**定義 5.1** (RAG A/B 測試): 對照實驗設計，比較不同 RAG 配置或算法在相同評估指標上的性能差異。

**統計假設檢驗**:
- **零假設 $H_0$**: $\mu_A = \mu_B$ (兩個版本性能無差異)
- **對立假設 $H_1$**: $\mu_A \neq \mu_B$ (存在顯著差異)

**功效分析**: 所需樣本量計算：

$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{(\mu_A - \mu_B)^2}$$

其中：
- $\alpha$: 第一類錯誤概率 (通常 0.05)
- $\beta$: 第二類錯誤概率 (通常 0.2)
- $\sigma$: 總體標準差
- $\mu_A - \mu_B$: 最小可檢測差異

#### **A/B 測試框架實現**

```python
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

class RAGABTestFramework:
    """RAG 系統 A/B 測試框架"""

    def __init__(self):
        self.experiments = {}  # 活躍實驗
        self.results_store = ExperimentResultStore()
        self.statistical_analyzer = StatisticalAnalyzer()

    async def create_experiment(self, experiment_config: Dict) -> str:
        """創建新的 A/B 測試實驗"""

        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 驗證實驗配置
        validation_result = await self._validate_experiment_config(experiment_config)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid experiment config: {validation_result['errors']}")

        # 計算所需樣本量
        required_sample_size = await self._calculate_required_sample_size(
            experiment_config
        )

        experiment = {
            "id": experiment_id,
            "name": experiment_config["name"],
            "description": experiment_config["description"],
            "variants": experiment_config["variants"],
            "success_metrics": experiment_config["success_metrics"],
            "traffic_allocation": experiment_config["traffic_allocation"],
            "required_sample_size": required_sample_size,
            "start_date": datetime.now(),
            "status": "active",
            "current_samples": {variant: 0 for variant in experiment_config["variants"]}
        }

        self.experiments[experiment_id] = experiment

        return experiment_id

    async def assign_user_to_variant(self, experiment_id: str,
                                   user_id: str) -> str:
        """分配用戶到實驗變體"""

        if experiment_id not in self.experiments:
            return "control"

        experiment = self.experiments[experiment_id]

        # 使用一致性哈希確保用戶總是分配到同一變體
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = hash(hash_input) % 10000
        allocation_value = hash_value / 10000.0

        # 根據流量分配確定變體
        cumulative_allocation = 0.0
        for variant, allocation in experiment["traffic_allocation"].items():
            cumulative_allocation += allocation
            if allocation_value <= cumulative_allocation:
                return variant

        return "control"  # 備用

    async def record_experiment_result(self, experiment_id: str,
                                     user_id: str, variant: str,
                                     query: str, result: Dict,
                                     user_feedback: Optional[Dict] = None):
        """記錄實驗結果"""

        if experiment_id not in self.experiments:
            return

        # 記錄實驗數據點
        data_point = {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant": variant,
            "timestamp": datetime.now(),
            "query": query,
            "result": result,
            "user_feedback": user_feedback,
            "session_metadata": {
                "processing_time": result.get("processing_time_ms", 0),
                "num_sources": len(result.get("sources", [])),
                "answer_length": len(result.get("answer", ""))
            }
        }

        await self.results_store.save_data_point(data_point)

        # 更新實驗樣本計數
        experiment = self.experiments[experiment_id]
        experiment["current_samples"][variant] += 1

        # 檢查是否達到統計顯著性
        if sum(experiment["current_samples"].values()) >= experiment["required_sample_size"]:
            await self._check_statistical_significance(experiment_id)

    async def analyze_experiment_results(self, experiment_id: str) -> Dict:
        """分析實驗結果"""

        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}

        experiment = self.experiments[experiment_id]

        # 獲取實驗數據
        experiment_data = await self.results_store.get_experiment_data(experiment_id)

        # 按變體分組數據
        variant_data = {}
        for data_point in experiment_data:
            variant = data_point["variant"]
            if variant not in variant_data:
                variant_data[variant] = []
            variant_data[variant].append(data_point)

        # 分析每個成功指標
        analysis_results = {}
        for metric in experiment["success_metrics"]:
            metric_analysis = await self._analyze_metric(variant_data, metric)
            analysis_results[metric] = metric_analysis

        # 統計顯著性檢驗
        significance_results = await self._perform_significance_tests(
            variant_data, experiment["success_metrics"]
        )

        # 生成實驗報告
        experiment_report = {
            "experiment_id": experiment_id,
            "experiment_name": experiment["name"],
            "analysis_timestamp": datetime.now(),
            "sample_sizes": {variant: len(data) for variant, data in variant_data.items()},
            "metric_analysis": analysis_results,
            "significance_tests": significance_results,
            "recommendation": await self._generate_experiment_recommendation(
                analysis_results, significance_results
            )
        }

        return experiment_report

    async def _analyze_metric(self, variant_data: Dict[str, List],
                            metric: str) -> Dict:
        """分析特定指標在不同變體間的表現"""

        metric_results = {}

        for variant, data_points in variant_data.items():
            metric_values = []

            for data_point in data_points:
                # 根據指標類型提取值
                if metric == "user_satisfaction":
                    feedback = data_point.get("user_feedback", {})
                    if feedback and "rating" in feedback:
                        metric_values.append(feedback["rating"])

                elif metric == "response_time":
                    time_ms = data_point["result"].get("processing_time_ms", 0)
                    metric_values.append(time_ms)

                elif metric == "answer_quality":
                    # 這裡需要實時品質評估
                    quality_score = await self._estimate_answer_quality(
                        data_point["query"],
                        data_point["result"].get("answer", "")
                    )
                    metric_values.append(quality_score)

            if metric_values:
                metric_results[variant] = {
                    "mean": np.mean(metric_values),
                    "std": np.std(metric_values),
                    "count": len(metric_values),
                    "median": np.median(metric_values),
                    "values": metric_values
                }

        return metric_results

    async def _perform_significance_tests(self, variant_data: Dict,
                                        metrics: List[str]) -> Dict:
        """執行統計顯著性檢驗"""

        significance_results = {}

        for metric in metrics:
            metric_analysis = await self._analyze_metric(variant_data, metric)

            if len(metric_analysis) >= 2:
                # 假設有 control 和 treatment 兩個變體
                variants = list(metric_analysis.keys())
                control_data = metric_analysis[variants[0]]["values"]
                treatment_data = metric_analysis[variants[1]]["values"]

                # 執行 t 檢驗
                t_stat, p_value = stats.ttest_ind(control_data, treatment_data)

                # 計算效應大小 (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(control_data) - 1) * np.var(control_data) +
                     (len(treatment_data) - 1) * np.var(treatment_data)) /
                    (len(control_data) + len(treatment_data) - 2)
                )

                effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std

                significance_results[metric] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "significant": p_value < 0.05,
                    "control_mean": np.mean(control_data),
                    "treatment_mean": np.mean(treatment_data),
                    "practical_significance": abs(effect_size) > 0.2  # Cohen's convention
                }

        return significance_results
```

---

## 6. 評估數據集構建

### 6.1 黃金標準數據集設計

#### **企業評估數據集的構建原則**

**原則 6.1** (代表性原則): 評估數據集應覆蓋企業實際使用中的各種查詢類型和難度分佈。

**數據集構建流程**:

```python
class EnterpriseEvaluationDatasetBuilder:
    """企業評估數據集構建器"""

    def __init__(self):
        self.query_categories = [
            "factual_lookup",      # 事實查詢
            "procedural_guide",    # 程序指南
            "analytical_complex",  # 分析型複雜查詢
            "troubleshooting",     # 故障排除
            "policy_compliance",   # 政策合規
            "multi_hop_reasoning"  # 多跳推理
        ]

        self.difficulty_levels = ["easy", "medium", "hard", "expert"]

    async def build_balanced_dataset(self, source_queries: List[Dict],
                                   target_size: int = 500) -> List[Dict]:
        """構建平衡的評估數據集"""

        # 目標分佈: 每個類別-難度組合的樣本數
        categories = len(self.query_categories)
        difficulties = len(self.difficulty_levels)
        samples_per_cell = target_size // (categories * difficulties)

        balanced_dataset = []

        # 對每個類別-難度組合採樣
        for category in self.query_categories:
            for difficulty in self.difficulty_levels:
                # 過濾符合條件的查詢
                matching_queries = [
                    q for q in source_queries
                    if (q.get("category") == category and
                        q.get("difficulty") == difficulty)
                ]

                # 採樣
                if len(matching_queries) >= samples_per_cell:
                    sampled = random.sample(matching_queries, samples_per_cell)
                else:
                    sampled = matching_queries
                    # 如果樣本不足，記錄警告
                    print(f"Warning: 不足樣本 {category}-{difficulty}: {len(matching_queries)}")

                balanced_dataset.extend(sampled)

        # 補充到目標大小
        remaining = target_size - len(balanced_dataset)
        if remaining > 0:
            unused_queries = [q for q in source_queries if q not in balanced_dataset]
            if unused_queries:
                additional_samples = random.sample(
                    unused_queries, min(remaining, len(unused_queries))
                )
                balanced_dataset.extend(additional_samples)

        # 打亂數據集順序
        random.shuffle(balanced_dataset)

        return balanced_dataset

    async def validate_dataset_quality(self, dataset: List[Dict]) -> Dict:
        """驗證數據集品質"""

        quality_metrics = {}

        # 1. 分佈平衡性檢查
        category_distribution = {}
        difficulty_distribution = {}

        for item in dataset:
            category = item.get("category", "unknown")
            difficulty = item.get("difficulty", "unknown")

            category_distribution[category] = category_distribution.get(category, 0) + 1
            difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1

        # 計算分佈熵 (越高越平衡)
        category_entropy = self._calculate_distribution_entropy(category_distribution)
        difficulty_entropy = self._calculate_distribution_entropy(difficulty_distribution)

        quality_metrics["category_balance"] = category_entropy / np.log(len(self.query_categories))
        quality_metrics["difficulty_balance"] = difficulty_entropy / np.log(len(self.difficulty_levels))

        # 2. 查詢品質檢查
        query_quality_scores = []
        for item in dataset:
            quality = await self._assess_query_quality(item)
            query_quality_scores.append(quality)

        quality_metrics["average_query_quality"] = np.mean(query_quality_scores)

        # 3. 答案品質檢查
        if all("expected_answer" in item for item in dataset):
            answer_quality_scores = []
            for item in dataset:
                answer_quality = await self._assess_answer_quality(
                    item["query"], item["expected_answer"]
                )
                answer_quality_scores.append(answer_quality)

            quality_metrics["average_answer_quality"] = np.mean(answer_quality_scores)

        return {
            "overall_quality_score": np.mean(list(quality_metrics.values())),
            "detailed_metrics": quality_metrics,
            "distributions": {
                "category": category_distribution,
                "difficulty": difficulty_distribution
            },
            "dataset_size": len(dataset),
            "quality_grade": self._assign_dataset_grade(np.mean(list(quality_metrics.values())))
        }

    def _calculate_distribution_entropy(self, distribution: Dict[str, int]) -> float:
        """計算分佈的信息熵"""

        total = sum(distribution.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log(probability)

        return entropy

    async def _assess_query_quality(self, query_item: Dict) -> float:
        """評估單個查詢的品質"""

        quality_score = 0.0

        query = query_item.get("query", "")

        # 1. 長度合理性
        query_length = len(query.split())
        if 3 <= query_length <= 50:
            quality_score += 0.2
        elif query_length > 50:
            quality_score += 0.1

        # 2. 語法正確性
        if await self._check_grammar(query):
            quality_score += 0.2

        # 3. 明確性
        ambiguity_score = await self._calculate_ambiguity(query)
        quality_score += 0.2 * (1.0 - ambiguity_score)

        # 4. 可回答性
        answerability = await self._assess_answerability(query)
        quality_score += 0.4 * answerability

        return quality_score
```

---

## 7. 本章總結與實踐指南

### 7.1 評估最佳實踐

#### **評估策略選擇指南**

| 評估目標 | 推薦指標 | 評估頻率 | 自動化程度 |
|---------|---------|---------|-----------|
| **系統調試** | Context Precision/Recall | 每次更新 | 完全自動化 |
| **質量保證** | Faithfulness, Answer Relevancy | 每日 | 完全自動化 |
| **用戶滿意度** | User Satisfaction, Task Success | 每週 | 半自動化 |
| **業務影響** | Cost per Query, ROI | 每月 | 人工分析 |

#### **評估工具鏈推薦**

**生產環境配置**:
```yaml
evaluation_stack:
  primary_framework: "RAGAS"
  monitoring_platform: "Opik + LangFuse"
  experimentation: "Custom A/B Testing"
  business_intelligence: "Streamlit + Plotly"

automation_level:
  component_testing: 100%
  system_testing: 90%
  business_evaluation: 60%
  strategic_assessment: 30%
```

### 7.2 持續改進循環

**改進循環模型**:

```
評估 → 分析 → 假設 → 實驗 → 驗證 → 部署 → 評估
```

每個循環週期建議為2-4週，確保快速迭代和持續優化。

---

**課程評估**: 本章內容在期末考試中占20%權重，重點考查評估框架設計和統計分析能力。

**項目要求**: 學生需完成一個完整的 RAG 系統評估項目，包括數據集構建、實驗設計和結果分析。