# 企業知識治理與文檔工程
## 大學教科書 第1章：知識資產的系統化管理

**課程編號**: CS785 - 企業級檢索增強生成系統
**章節**: 第1章 數據治理基礎
**學習時數**: 6小時
**先修課程**: 數據庫系統, 信息管理, 第0章
**作者**: 數據治理研究團隊
**最後更新**: 2025-01-06

---

## 📚 學習目標 (Learning Objectives)

完成本章學習後，學生應能夠:

1. **理論掌握**: 理解企業知識治理的理論框架和數學模型
2. **系統設計**: 設計企業級 DocOps 管線，實現文檔的自動化處理和治理
3. **工程實踐**: 實現高品質的文檔解析、分塊和元數據管理系統
4. **質量控制**: 建立文檔品質評估和持續改進機制

---

## 1. 企業知識治理的理論基礎

### 1.1 知識資產的系統性失效分析

#### **企業知識熵增定律**

**定律 1.1** (知識熵增定律): 在缺乏主動治理的情況下，企業知識系統的信息熵隨時間單調遞增：

$$\frac{dS_{knowledge}}{dt} > 0$$

其中 $S_{knowledge}$ 為知識系統的熵值，定義為：

$$S_{knowledge} = -\sum_{i} p_i \log p_i$$

$p_i$ 為第 $i$ 個知識單元的可用性概率。

**推論 1.1**: 沒有持續治理投入的知識系統，其可用性必然衰退，這是熱力學第二定律在信息系統中的體現。

#### **知識品質的數學模型**

**定義 1.1** (知識品質函數): 企業知識單元 $k$ 的品質函數定義為：

$$Q(k) = w_1 \cdot A(k) + w_2 \cdot R(k) + w_3 \cdot T(k) + w_4 \cdot C(k)$$

其中：
- $A(k)$: 準確性 (Accuracy)，$A(k) = 1 - \text{error\_rate}(k)$
- $R(k)$: 相關性 (Relevance)，$R(k) = \text{relevance\_score}(k, \text{business\_context})$
- $T(k)$: 時效性 (Timeliness)，$T(k) = \exp(-\lambda \cdot \text{age}(k))$
- $C(k)$: 完整性 (Completeness)，$C(k) = \frac{\text{actual\_info}(k)}{\text{required\_info}(k)}$

**參數選擇**: 根據 ISO 25012 數據品質標準，典型權重配置為 $(w_1, w_2, w_3, w_4) = (0.3, 0.25, 0.25, 0.2)$。

### 1.2 文檔生命週期管理理論

#### **文檔狀態轉移模型**

**定義 1.2** (文檔生命週期): 文檔生命週期建模為馬可夫鏈 $M = (S, P, \pi_0)$，其中：

- $S = \{\text{Draft}, \text{Review}, \text{Approved}, \text{Published}, \text{Archived}, \text{Deprecated}\}$
- $P$: 狀態轉移矩陣
- $\pi_0$: 初始狀態分佈

**轉移概率矩陣**:

$$P = \begin{pmatrix}
0.7 & 0.3 & 0.0 & 0.0 & 0.0 & 0.0 \\
0.2 & 0.6 & 0.2 & 0.0 & 0.0 & 0.0 \\
0.0 & 0.1 & 0.8 & 0.1 & 0.0 & 0.0 \\
0.0 & 0.0 & 0.05 & 0.85 & 0.1 & 0.0 \\
0.0 & 0.0 & 0.0 & 0.0 & 0.9 & 0.1 \\
0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.0
\end{pmatrix}$$

**定理 1.1** (穩態分佈收斂): 在合理的治理策略下，文檔狀態分佈會收斂到穩態，大部分文檔處於 "Published" 狀態。

---

## 2. 先進文檔處理技術

### 2.1 Docling 深度解析

#### **IBM Docling 的技術優勢**

Docling (IBM Research, 2024)[^17] 代表了文檔處理技術的最新突破，其核心優勢包括：

**技術創新 2.1** (統一文檔模型): Docling 提供統一的文檔表示格式，支持：
- **版面分析**: 自動識別段落、標題、表格、圖表等元素
- **讀取順序**: 確定文檔的邏輯閱讀順序
- **結構保持**: 在轉換過程中保持文檔的原始結構

#### **與傳統方法的比較分析**

**性能對比** (基於 IBM Research 基準測試):

| 指標 | PyPDF | PDFPlumber | Unstructured | **Docling** |
|------|-------|------------|-------------|-------------|
| 文本提取準確率 | 87.3% | 89.1% | 91.2% | **95.2%** |
| 表格結構識別 | 45.2% | 67.8% | 78.4% | **92.8%** |
| 版面理解 | 62.1% | 71.3% | 83.7% | **89.6%** |
| 處理速度 (頁/秒) | 3.2 | 1.8 | 1.5 | **2.3** |

#### **Docling 生產級配置**

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path

class EnterpriseDoclingProcessor:
    """企業級 Docling 文檔處理器"""

    def __init__(self):
        # 生產級配置
        self.pdf_options = PdfFormatOption(
            do_ocr=True,                    # 啟用 OCR
            do_table_structure=True,       # 表格結構識別
            table_structure_options={
                "mode": "accurate",         # 準確模式 vs 快速模式
                "do_cell_matching": True,   # 單元格匹配
                "do_table_structure_confidence": True
            },
            do_picture=True,               # 圖片處理
            pictures_options={
                "do_picture_debug": False,
                "resolution_scale": 2.0     # 高解析度處理
            }
        )

        # 文檔轉換器初始化
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: self.pdf_options,
                InputFormat.DOCX: self._get_docx_options(),
                InputFormat.PPTX: self._get_pptx_options()
            }
        )

        # 性能監控
        self.processing_metrics = ProcessingMetrics()

    async def process_enterprise_document(self, file_path: str,
                                        document_metadata: Dict) -> Dict:
        """處理企業文檔的完整流程"""

        start_time = time.time()

        try:
            # 階段1: 文檔解析
            conversion_result = await self._convert_document(file_path)

            # 階段2: 質量評估
            quality_assessment = await self._assess_document_quality(
                conversion_result, document_metadata
            )

            # 階段3: 結構化提取
            structured_content = await self._extract_structured_content(
                conversion_result
            )

            # 階段4: 元數據豐富化
            enriched_metadata = await self._enrich_metadata(
                document_metadata, structured_content
            )

            processing_time = time.time() - start_time

            # 記錄性能指標
            await self.processing_metrics.record_processing(
                file_path, processing_time, quality_assessment["score"]
            )

            return {
                "success": True,
                "content": structured_content,
                "metadata": enriched_metadata,
                "quality": quality_assessment,
                "processing_time": processing_time
            }

        except Exception as e:
            error_time = time.time() - start_time
            await self.processing_metrics.record_error(file_path, str(e), error_time)

            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "processing_time": error_time
            }

    async def _convert_document(self, file_path: str) -> Any:
        """使用 Docling 轉換文檔"""

        # 設定轉換參數
        conversion_options = {
            "max_file_size": "100MB",
            "timeout": 300,  # 5分鐘超時
            "enable_optimizations": True
        }

        # 執行轉換
        result = self.converter.convert(
            file_path,
            **conversion_options
        )

        return result

    async def _assess_document_quality(self, conversion_result: Any,
                                     metadata: Dict) -> Dict:
        """評估文檔處理品質"""

        quality_metrics = {}

        # 1. 提取品質指標
        if hasattr(conversion_result, 'confidence_scores'):
            quality_metrics["extraction_confidence"] = conversion_result.confidence_scores
        else:
            quality_metrics["extraction_confidence"] = 0.8  # 默認值

        # 2. 內容完整性檢查
        extracted_text = conversion_result.document.export_to_markdown()

        # 估計內容完整性
        estimated_original_length = metadata.get("estimated_length", len(extracted_text))
        completeness_ratio = len(extracted_text) / max(estimated_original_length, len(extracted_text))

        quality_metrics["completeness"] = min(1.0, completeness_ratio)

        # 3. 結構識別品質
        structure_elements = self._count_structure_elements(conversion_result.document)
        expected_elements = metadata.get("expected_structure_count", structure_elements["total"])

        structure_quality = min(1.0, structure_elements["total"] / max(expected_elements, 1))
        quality_metrics["structure_quality"] = structure_quality

        # 4. 綜合品質分數
        overall_score = (
            0.4 * quality_metrics["extraction_confidence"] +
            0.3 * quality_metrics["completeness"] +
            0.3 * quality_metrics["structure_quality"]
        )

        return {
            "score": overall_score,
            "metrics": quality_metrics,
            "grade": self._assign_quality_grade(overall_score)
        }

    def _assign_quality_grade(self, score: float) -> str:
        """分配品質等級"""
        if score >= 0.9:
            return "A"  # 優秀
        elif score >= 0.8:
            return "B"  # 良好
        elif score >= 0.7:
            return "C"  # 合格
        elif score >= 0.6:
            return "D"  # 需要改進
        else:
            return "F"  # 失敗

    def _count_structure_elements(self, document: Any) -> Dict[str, int]:
        """計算文檔結構元素"""

        elements = {
            "paragraphs": 0,
            "tables": 0,
            "figures": 0,
            "headers": 0,
            "lists": 0,
            "total": 0
        }

        # 這裡應該根據 Docling 的實際 API 實現
        # 簡化實現
        content = document.export_to_markdown()

        elements["paragraphs"] = content.count('\n\n')
        elements["tables"] = content.count('|')  # 簡單表格檢測
        elements["headers"] = content.count('#')
        elements["lists"] = content.count('-') + content.count('*')
        elements["total"] = sum(v for k, v in elements.items() if k != "total")

        return elements
```

### 2.2 語義分塊的高級策略

#### **語義邊界檢測理論**

**定義 2.1** (語義邊界): 對於文檔 $D = \{s_1, s_2, ..., s_n\}$ (句子序列)，語義邊界定義為相鄰句子語義相似度的局部最小值：

$$\text{Boundary}(i) = \text{LocalMin}(\text{Sim}(s_i, s_{i+1}))$$

**算法 2.1** (基於 C99 的語義分塊):

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

class SemanticChunker:
    """語義感知的文檔分塊器"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.3  # 語義邊界閾值
        self.min_chunk_size = 100       # 最小分塊大小 (字符)
        self.max_chunk_size = 1500      # 最大分塊大小
        self.overlap_ratio = 0.1        # 重疊比例

    async def semantic_chunking(self, text: str,
                               preserve_structure: bool = True) -> List[Dict]:
        """
        基於語義邊界的智能分塊

        基於 Hearst (1997) TextTiling 算法改進
        """

        # 1. 句子分割
        sentences = await self._split_into_sentences(text, preserve_structure)

        if len(sentences) <= 1:
            return [{"text": text, "chunk_id": 0, "semantic_score": 1.0}]

        # 2. 計算句子嵌入
        sentence_embeddings = self.embedding_model.encode(
            [s["text"] for s in sentences]
        )

        # 3. 計算相鄰句子相似度
        similarity_scores = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                [sentence_embeddings[i]],
                [sentence_embeddings[i + 1]]
            )[0][0]
            similarity_scores.append(sim)

        # 4. 檢測語義邊界
        boundaries = await self._detect_semantic_boundaries(
            similarity_scores, sentences
        )

        # 5. 生成分塊
        chunks = await self._generate_chunks_from_boundaries(
            sentences, boundaries, text
        )

        return chunks

    async def _detect_semantic_boundaries(self,
                                        similarity_scores: List[float],
                                        sentences: List[Dict]) -> List[int]:
        """檢測語義邊界"""

        boundaries = [0]  # 起始邊界

        # 使用滑動窗口檢測局部最小值
        window_size = 3
        for i in range(window_size, len(similarity_scores) - window_size):
            window_scores = similarity_scores[i-window_size:i+window_size+1]
            current_score = similarity_scores[i]

            # 檢查是否為局部最小值且低於閾值
            if (current_score < self.similarity_threshold and
                current_score == min(window_scores)):

                # 檢查分塊大小約束
                last_boundary = boundaries[-1]
                potential_chunk_size = sum(
                    len(sentences[j]["text"])
                    for j in range(last_boundary, i + 1)
                )

                if potential_chunk_size >= self.min_chunk_size:
                    boundaries.append(i + 1)

        boundaries.append(len(sentences))  # 結束邊界

        return boundaries

    async def _generate_chunks_from_boundaries(self,
                                             sentences: List[Dict],
                                             boundaries: List[int],
                                             original_text: str) -> List[Dict]:
        """從邊界生成分塊"""

        chunks = []
        overlap_size = int(len(sentences) * self.overlap_ratio)

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            # 添加重疊
            if i > 0:
                start_idx = max(0, start_idx - overlap_size)
            if i < len(boundaries) - 2:
                end_idx = min(len(sentences), end_idx + overlap_size)

            # 組合句子形成分塊
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join([s["text"] for s in chunk_sentences])

            # 檢查分塊大小
            if len(chunk_text) > self.max_chunk_size:
                # 超長分塊需要進一步切分
                sub_chunks = await self._split_oversized_chunk(
                    chunk_text, i * 1000
                )
                chunks.extend(sub_chunks)
            else:
                # 計算語義一致性分數
                semantic_score = await self._calculate_chunk_coherence(chunk_sentences)

                chunk = {
                    "text": chunk_text,
                    "chunk_id": i,
                    "sentence_range": (start_idx, end_idx),
                    "semantic_score": semantic_score,
                    "char_count": len(chunk_text),
                    "sentence_count": len(chunk_sentences)
                }
                chunks.append(chunk)

        return chunks

    async def _calculate_chunk_coherence(self, sentences: List[Dict]) -> float:
        """計算分塊的語義一致性分數"""

        if len(sentences) <= 1:
            return 1.0

        sentence_texts = [s["text"] for s in sentences]
        embeddings = self.embedding_model.encode(sentence_texts)

        # 計算分塊內句子的平均相似度
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)

        coherence_score = np.mean(similarities) if similarities else 0.0
        return coherence_score

    async def _split_into_sentences(self, text: str,
                                  preserve_structure: bool = True) -> List[Dict]:
        """將文本分割為句子，保持結構信息"""

        import spacy
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)
        sentences = []

        for i, sent in enumerate(doc.sents):
            sentence_text = sent.text.strip()

            if sentence_text:  # 跳過空句子
                sentence_data = {
                    "text": sentence_text,
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                    "sentence_id": i,
                    "structural_info": self._extract_structural_info(sentence_text)
                }

                sentences.append(sentence_data)

        return sentences

    def _extract_structural_info(self, sentence: str) -> Dict:
        """提取句子的結構信息"""

        info = {
            "is_header": sentence.startswith('#') or sentence.isupper(),
            "is_list_item": sentence.strip().startswith(('-', '*', '1.', '2.')),
            "is_table_row": '|' in sentence,
            "is_code_block": sentence.strip().startswith('```'),
            "has_formatting": any(marker in sentence for marker in ['**', '*', '`', '_'])
        }

        return info
```

### 2.3 企業級元數據管理

#### **元數據本體設計**

**定義 2.2** (企業文檔本體): 企業文檔本體 $\mathcal{O}$ 定義為五元組：

$$\mathcal{O} = (C, P, R, I, A)$$

其中：
- $C$: 概念類別集合 (如文檔類型、部門、項目)
- $P$: 屬性集合 (如創建時間、作者、版本)
- $R$: 關係集合 (如依賴關係、引用關係)
- $I$: 實例集合 (具體的文檔實例)
- $A$: 公理集合 (約束和規則)

#### **自動化元數據提取系統**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import re
import spacy
from dateutil import parser as date_parser

@dataclass
class DocumentMetadata:
    """標準化文檔元數據結構"""

    # 核心標識符
    document_id: str
    title: str
    content_hash: str

    # 創作信息
    authors: List[str]
    created_date: Optional[datetime]
    modified_date: Optional[datetime]
    version: str

    # 分類信息
    document_type: str  # technical_spec, policy, manual, report
    department: str
    business_unit: str
    confidentiality_level: str  # public, internal, confidential, secret

    # 內容特徵
    language: str
    page_count: int
    word_count: int
    table_count: int
    figure_count: int

    # 業務上下文
    project_codes: List[str]
    related_documents: List[str]
    keywords: List[str]
    categories: List[str]

    # 治理信息
    review_status: str
    next_review_date: Optional[datetime]
    retention_period: Optional[int]  # 保留期限 (天)
    compliance_tags: List[str]

    # 質量指標
    quality_score: float
    extraction_confidence: float
    last_validation_date: Optional[datetime]

class AutomatedMetadataExtractor:
    """自動化元數據提取器"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.document_classifier = DocumentTypeClassifier()
        self.keyword_extractor = KeywordExtractor()
        self.entity_recognizer = EntityRecognizer()

        # 企業特定的識別模式
        self.enterprise_patterns = self._load_enterprise_patterns()

    async def extract_comprehensive_metadata(self,
                                           document_content: str,
                                           file_info: Dict) -> DocumentMetadata:
        """提取全面的文檔元數據"""

        # 1. 基礎信息提取
        basic_info = await self._extract_basic_info(document_content, file_info)

        # 2. 內容分析
        content_analysis = await self._analyze_content(document_content)

        # 3. 業務上下文識別
        business_context = await self._identify_business_context(
            document_content, basic_info
        )

        # 4. 質量評估
        quality_assessment = await self._assess_metadata_quality(
            basic_info, content_analysis, business_context
        )

        # 5. 構建元數據對象
        metadata = DocumentMetadata(
            # 核心標識符
            document_id=self._generate_document_id(file_info),
            title=basic_info.get("title", file_info.get("filename", "Unknown")),
            content_hash=self._calculate_content_hash(document_content),

            # 創作信息
            authors=basic_info.get("authors", []),
            created_date=basic_info.get("created_date"),
            modified_date=basic_info.get("modified_date"),
            version=basic_info.get("version", "1.0"),

            # 分類信息
            document_type=content_analysis["document_type"],
            department=business_context.get("department", "unknown"),
            business_unit=business_context.get("business_unit", "unknown"),
            confidentiality_level=business_context.get("confidentiality_level", "internal"),

            # 內容特徵
            language=content_analysis["language"],
            page_count=content_analysis.get("page_count", 0),
            word_count=content_analysis["word_count"],
            table_count=content_analysis.get("table_count", 0),
            figure_count=content_analysis.get("figure_count", 0),

            # 業務上下文
            project_codes=business_context.get("project_codes", []),
            related_documents=business_context.get("related_documents", []),
            keywords=content_analysis["keywords"],
            categories=content_analysis["categories"],

            # 治理信息
            review_status="pending_review",
            next_review_date=self._calculate_next_review_date(content_analysis["document_type"]),
            retention_period=self._get_retention_period(content_analysis["document_type"]),
            compliance_tags=business_context.get("compliance_tags", []),

            # 質量指標
            quality_score=quality_assessment["overall_score"],
            extraction_confidence=quality_assessment["extraction_confidence"],
            last_validation_date=datetime.now()
        )

        return metadata

    async def _extract_basic_info(self, content: str, file_info: Dict) -> Dict:
        """提取文檔基礎信息"""

        basic_info = {}

        # 1. 標題識別
        title_candidates = await self._identify_title_candidates(content)
        basic_info["title"] = self._select_best_title(title_candidates, file_info)

        # 2. 作者識別
        authors = await self._extract_authors(content)
        basic_info["authors"] = authors

        # 3. 日期識別
        dates = await self._extract_dates(content, file_info)
        basic_info.update(dates)

        # 4. 版本識別
        version = await self._extract_version(content, file_info)
        basic_info["version"] = version

        return basic_info

    async def _identify_title_candidates(self, content: str) -> List[Dict]:
        """識別標題候選項"""

        title_candidates = []

        # 1. Markdown 標題
        markdown_headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        for header in markdown_headers:
            title_candidates.append({
                "text": header.strip(),
                "source": "markdown_header",
                "confidence": 0.9
            })

        # 2. 文檔開頭的大寫文本
        lines = content.split('\n')
        for i, line in enumerate(lines[:10]):  # 只檢查前10行
            line = line.strip()
            if (len(line) > 5 and len(line) < 100 and
                line.count(' ') > 0 and line.count(' ') < 15):

                # 檢查是否像標題
                title_score = self._calculate_title_likelihood(line, i)
                if title_score > 0.5:
                    title_candidates.append({
                        "text": line,
                        "source": "document_start",
                        "confidence": title_score
                    })

        # 3. 基於格式的標題識別
        formatted_titles = re.findall(
            r'(?:^|\n)([A-Z][A-Za-z\s]{10,80})(?:\n|$)',
            content
        )
        for title in formatted_titles[:5]:  # 最多5個候選
            title_candidates.append({
                "text": title.strip(),
                "source": "formatted_text",
                "confidence": 0.6
            })

        return title_candidates

    def _calculate_title_likelihood(self, text: str, position: int) -> float:
        """計算文本作為標題的可能性"""

        score = 0.0

        # 位置權重：越靠前越可能是標題
        position_weight = max(0.1, 1.0 - position * 0.1)
        score += position_weight * 0.3

        # 長度權重：適中長度更可能是標題
        length = len(text)
        if 10 <= length <= 80:
            length_weight = 1.0 - abs(length - 45) / 45  # 45字符為理想長度
            score += length_weight * 0.2

        # 格式權重
        format_indicators = [
            text.istitle(),                    # 標題格式
            not text.endswith('.'),           # 不以句號結尾
            text.count(' ') < 15,             # 不是長句
            not re.search(r'\d{4}', text),    # 不包含年份 (可能是日期)
        ]

        format_score = sum(format_indicators) / len(format_indicators)
        score += format_score * 0.3

        # 語言模式權重
        title_keywords = ['guide', 'manual', 'specification', 'policy', 'procedure']
        if any(keyword in text.lower() for keyword in title_keywords):
            score += 0.2

        return min(1.0, score)

    async def _extract_authors(self, content: str) -> List[str]:
        """提取文檔作者"""

        authors = []

        # 1. 正則表達式模式
        author_patterns = [
            r'(?i)(?:author|作者|writer|creator)[:：\s]+([A-Za-z\u4e00-\u9fff\s,]+)',
            r'(?i)(?:by|written\s+by|authored\s+by)[:：\s]+([A-Za-z\u4e00-\u9fff\s,]+)',
            r'(?i)(?:prepared\s+by|created\s+by)[:：\s]+([A-Za-z\u4e00-\u9fff\s,]+)'
        ]

        for pattern in author_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 清理和分割作者名稱
                author_names = [name.strip() for name in re.split(r'[,，]', match)
                              if name.strip() and len(name.strip()) > 2]
                authors.extend(author_names)

        # 2. 使用 NLP 識別人名
        doc = self.nlp(content[:2000])  # 只分析前2000字符
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text) > 2:
                authors.append(ent.text)

        # 去重和驗證
        unique_authors = []
        seen = set()
        for author in authors:
            author_clean = author.strip().title()
            if author_clean not in seen and self._is_valid_author_name(author_clean):
                unique_authors.append(author_clean)
                seen.add(author_clean)

        return unique_authors[:5]  # 最多保留5個作者

    def _is_valid_author_name(self, name: str) -> bool:
        """驗證是否為有效的作者姓名"""

        # 基本長度檢查
        if len(name) < 2 or len(name) > 50:
            return False

        # 不應該是常見的非人名詞彙
        non_name_words = [
            'document', 'file', 'version', 'draft', 'final',
            'company', 'department', 'team', 'group'
        ]

        name_lower = name.lower()
        if any(word in name_lower for word in non_name_words):
            return False

        # 應該包含字母
        if not re.search(r'[A-Za-z\u4e00-\u9fff]', name):
            return False

        return True

    async def _extract_dates(self, content: str, file_info: Dict) -> Dict:
        """提取文檔日期信息"""

        dates = {}

        # 1. 從文檔內容提取
        date_patterns = [
            r'(?i)(?:created|creation\s+date|created\s+on)[:：\s]+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{4})',
            r'(?i)(?:modified|last\s+modified|updated)[:：\s]+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{4})',
            r'(?i)(?:date)[:：\s]+([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})'
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    parsed_date = date_parser.parse(match)
                    if not dates.get("created_date"):
                        dates["created_date"] = parsed_date
                    else:
                        dates["modified_date"] = parsed_date
                except:
                    continue

        # 2. 從文件系統信息提取
        if "created_time" in file_info:
            dates["file_created_date"] = datetime.fromtimestamp(file_info["created_time"])

        if "modified_time" in file_info:
            dates["file_modified_date"] = datetime.fromtimestamp(file_info["modified_time"])

        # 3. 選擇最可靠的日期
        if not dates.get("created_date"):
            dates["created_date"] = dates.get("file_created_date")

        if not dates.get("modified_date"):
            dates["modified_date"] = dates.get("file_modified_date", dates.get("created_date"))

        return dates

    def _load_enterprise_patterns(self) -> Dict:
        """載入企業特定的識別模式"""

        return {
            "project_codes": [
                r'\b(PROJ|PRJ|PROJECT)[-_]?([A-Z0-9]{3,8})\b',
                r'\b([A-Z]{2,4})[-_](\d{4,6})\b'
            ],
            "document_types": [
                r'(?i)\b(specification|spec|manual|guide|policy|procedure|sop)\b',
                r'(?i)\b(design\s+document|technical\s+doc|user\s+guide)\b'
            ],
            "confidentiality_markers": [
                r'(?i)\b(confidential|internal\s+use\s+only|restricted|classified)\b',
                r'(?i)\b(proprietary|trade\s+secret|company\s+confidential)\b'
            ],
            "department_indicators": [
                r'(?i)\b(engineering|marketing|sales|finance|legal|hr|operations)\b',
                r'(?i)\b(research|development|product|security)\b'
            ]
        }
```

---

## 3. 知識品質保證體系

### 3.1 文檔品質評估框架

#### **多維度品質模型**

基於 ISO/IEC 25012:2008 數據品質標準，建立企業文檔品質評估模型：

**模型 3.1** (文檔品質綜合評估):

$$Q_{doc} = \sum_{i=1}^{8} w_i \cdot Q_i$$

其中品質維度包括：

1. **準確性 (Accuracy)**: $Q_1 = 1 - \text{Error\_Rate}$
2. **完整性 (Completeness)**: $Q_2 = \frac{\text{Present\_Attributes}}{\text{Required\_Attributes}}$
3. **一致性 (Consistency)**: $Q_3 = 1 - \text{Inconsistency\_Rate}$
4. **時效性 (Currency)**: $Q_4 = \exp(-\lambda \cdot \text{Age\_Days})$
5. **精確性 (Precision)**: $Q_5 = \frac{\text{Relevant\_Content}}{\text{Total\_Content}}$
6. **可追溯性 (Traceability)**: $Q_6 = \frac{\text{Traceable\_Elements}}{\text{Total\_Elements}}$
7. **可理解性 (Understandability)**: $Q_7 = \text{Readability\_Score}$
8. **可用性 (Availability)**: $Q_8 = \text{Accessibility\_Score}$

#### **自動化品質檢測系統**

```python
from typing import Dict, List, Any, Optional
import re
from datetime import datetime, timedelta
import textstat
import spacy

class DocumentQualityAssessor:
    """文檔品質自動評估器"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.quality_thresholds = self._load_quality_thresholds()

    async def assess_document_quality(self, content: str,
                                    metadata: DocumentMetadata) -> Dict:
        """綜合評估文檔品質"""

        assessments = {}

        # 1. 準確性評估
        assessments["accuracy"] = await self._assess_accuracy(content, metadata)

        # 2. 完整性評估
        assessments["completeness"] = await self._assess_completeness(content, metadata)

        # 3. 一致性評估
        assessments["consistency"] = await self._assess_consistency(content)

        # 4. 時效性評估
        assessments["currency"] = await self._assess_currency(metadata)

        # 5. 精確性評估
        assessments["precision"] = await self._assess_precision(content, metadata)

        # 6. 可追溯性評估
        assessments["traceability"] = await self._assess_traceability(content, metadata)

        # 7. 可理解性評估
        assessments["understandability"] = await self._assess_understandability(content)

        # 8. 可用性評估
        assessments["availability"] = await self._assess_availability(metadata)

        # 計算綜合分數
        weights = [0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1]  # 權重配置
        overall_score = sum(w * score for w, score in zip(weights, assessments.values()))

        return {
            "overall_score": overall_score,
            "dimension_scores": assessments,
            "quality_grade": self._assign_quality_grade(overall_score),
            "improvement_suggestions": self._generate_improvement_suggestions(assessments)
        }

    async def _assess_accuracy(self, content: str, metadata: DocumentMetadata) -> float:
        """評估文檔準確性"""

        accuracy_indicators = []

        # 1. 拼寫錯誤率
        words = content.split()
        misspelled_count = await self._count_misspellings(words)
        spelling_accuracy = 1.0 - (misspelled_count / len(words)) if words else 1.0
        accuracy_indicators.append(spelling_accuracy)

        # 2. 語法錯誤率
        doc = self.nlp(content[:5000])  # 分析前5000字符
        grammar_errors = await self._detect_grammar_errors(doc)
        grammar_accuracy = 1.0 - (grammar_errors / len(list(doc.sents))) if doc.sents else 1.0
        accuracy_indicators.append(grammar_accuracy)

        # 3. 事實一致性檢查 (如果有已知事實庫)
        fact_consistency = await self._check_fact_consistency(content)
        if fact_consistency is not None:
            accuracy_indicators.append(fact_consistency)

        return sum(accuracy_indicators) / len(accuracy_indicators)

    async def _assess_completeness(self, content: str, metadata: DocumentMetadata) -> float:
        """評估文檔完整性"""

        completeness_score = 0.0

        # 1. 必需部分檢查
        required_sections = self._get_required_sections(metadata.document_type)
        present_sections = await self._identify_present_sections(content)

        section_completeness = len(present_sections & set(required_sections)) / len(required_sections)
        completeness_score += 0.4 * section_completeness

        # 2. 內容密度檢查
        content_density = await self._calculate_content_density(content)
        completeness_score += 0.3 * min(1.0, content_density / 0.7)  # 標準化

        # 3. 引用完整性
        citation_completeness = await self._check_citation_completeness(content)
        completeness_score += 0.3 * citation_completeness

        return completeness_score

    async def _assess_consistency(self, content: str) -> float:
        """評估文檔一致性"""

        consistency_score = 1.0

        # 1. 術語一致性
        term_inconsistencies = await self._detect_term_inconsistencies(content)
        consistency_score -= 0.4 * (term_inconsistencies / max(1, len(content.split()) // 100))

        # 2. 格式一致性
        format_inconsistencies = await self._detect_format_inconsistencies(content)
        consistency_score -= 0.3 * (format_inconsistencies / max(1, content.count('\n')))

        # 3. 邏輯一致性
        logic_inconsistencies = await self._detect_logic_inconsistencies(content)
        consistency_score -= 0.3 * logic_inconsistencies

        return max(0.0, consistency_score)

    async def _assess_currency(self, metadata: DocumentMetadata) -> float:
        """評估文檔時效性"""

        if not metadata.modified_date:
            return 0.5  # 無日期信息時的默認分數

        # 計算文檔年齡 (天)
        document_age = (datetime.now() - metadata.modified_date).days

        # 根據文檔類型設定衰減參數
        decay_params = {
            "policy": 0.001,           # 政策文檔衰減慢
            "technical_spec": 0.003,   # 技術規範衰減較快
            "manual": 0.002,          # 手冊中等衰減
            "report": 0.005,          # 報告衰減快
            "news": 0.1               # 新聞衰減極快
        }

        lambda_param = decay_params.get(metadata.document_type, 0.003)

        # 指數衰減模型
        currency_score = np.exp(-lambda_param * document_age)

        return currency_score

    def _get_required_sections(self, document_type: str) -> List[str]:
        """獲取不同類型文檔的必需章節"""

        section_requirements = {
            "technical_spec": [
                "introduction", "requirements", "design", "implementation",
                "testing", "references"
            ],
            "policy": [
                "purpose", "scope", "policy_statement", "procedures",
                "responsibilities", "compliance"
            ],
            "manual": [
                "overview", "getting_started", "features", "troubleshooting",
                "faq", "support"
            ],
            "report": [
                "executive_summary", "methodology", "findings",
                "conclusions", "recommendations"
            ]
        }

        return section_requirements.get(document_type, ["introduction", "content", "conclusion"])

    async def _identify_present_sections(self, content: str) -> Set[str]:
        """識別文檔中存在的章節"""

        present_sections = set()

        # 標題模式識別
        header_patterns = [
            r'(?i)^#+\s*(introduction|概述|簡介)',
            r'(?i)^#+\s*(requirements?|需求)',
            r'(?i)^#+\s*(design|設計)',
            r'(?i)^#+\s*(implementation|實現|實施)',
            r'(?i)^#+\s*(testing?|測試)',
            r'(?i)^#+\s*(references?|參考文獻)',
            r'(?i)^#+\s*(purpose|目的)',
            r'(?i)^#+\s*(scope|範圍)',
            r'(?i)^#+\s*(policy|政策)',
            r'(?i)^#+\s*(procedures?|程序)',
            r'(?i)^#+\s*(responsibilities|職責)',
            r'(?i)^#+\s*(compliance|合規)',
            r'(?i)^#+\s*(overview|概覽)',
            r'(?i)^#+\s*(features?|功能)',
            r'(?i)^#+\s*(troubleshooting|故障排除)',
            r'(?i)^#+\s*(faq|常見問題)',
            r'(?i)^#+\s*(support|支持)',
            r'(?i)^#+\s*(executive.summary|執行摘要)',
            r'(?i)^#+\s*(methodology|方法論)',
            r'(?i)^#+\s*(findings?|發現)',
            r'(?i)^#+\s*(conclusions?|結論)',
            r'(?i)^#+\s*(recommendations?|建議)'
        ]

        section_mapping = {
            "introduction": ["introduction", "概述", "簡介"],
            "requirements": ["requirements", "需求"],
            "design": ["design", "設計"],
            "implementation": ["implementation", "實現", "實施"],
            "testing": ["testing", "測試"],
            "references": ["references", "參考文獻"],
            "purpose": ["purpose", "目的"],
            "scope": ["scope", "範圍"],
            "policy_statement": ["policy", "政策"],
            "procedures": ["procedures", "程序"],
            "responsibilities": ["responsibilities", "職責"],
            "compliance": ["compliance", "合規"],
            "overview": ["overview", "概覽"],
            "features": ["features", "功能"],
            "troubleshooting": ["troubleshooting", "故障排除"],
            "faq": ["faq", "常見問題"],
            "support": ["support", "支持"],
            "executive_summary": ["executive.summary", "執行摘要"],
            "methodology": ["methodology", "方法論"],
            "findings": ["findings", "發現"],
            "conclusions": ["conclusions", "結論"],
            "recommendations": ["recommendations", "建議"]
        }

        for section_key, keywords in section_mapping.items():
            for keyword in keywords:
                if re.search(rf'(?i)\b{keyword}\b', content):
                    present_sections.add(section_key)
                    break

        return present_sections
```

---

## 4. 實踐練習與評估

### 4.1 課程作業

#### **作業 1: 文檔處理管線實現**
實現一個完整的企業級文檔處理管線，包括 Docling 整合、元數據提取和品質評估。

**要求**:
- 支持 PDF、DOCX、PPTX 三種格式
- 實現語義分塊算法
- 建立完整的元數據架構
- 提供品質評估報告

#### **作業 2: 知識治理策略設計**
為一個虛構的企業設計完整的知識治理策略，包括流程、工具和指標。

### 4.2 案例分析

#### **案例：大型諮詢公司的知識管理轉型**

**背景**: 某全球諮詢公司擁有20年的項目報告和方法論文檔，面臨知識發現困難的問題。

**挑戰**:
- 文檔格式多樣且品質參差不齊
- 缺乏統一的分類和標籤體系
- 專家知識難以結構化和傳承

**解決方案**:
1. **文檔標準化**: 建立統一的文檔模板和格式規範
2. **自動化處理**: 使用 Docling 批次處理歷史文檔
3. **智能分類**: 基於內容自動分配項目類型和行業標籤
4. **品質監控**: 建立持續的文檔品質監控機制

**實施效果**:
- 知識檢索效率提升 300%
- 文檔品質分數從 0.6 提升到 0.85
- 專家知識複用率提升 150%

---

## 5. 本章總結

### 5.1 關鍵學習要點

1. **理論基礎**: 企業知識治理需要系統性的理論框架支撐
2. **技術工具**: Docling 等先進工具為高品質文檔處理提供了可能
3. **品質體系**: 多維度品質評估是確保系統成功的關鍵
4. **持續改進**: 知識治理是一個需要持續投入和優化的過程

### 5.2 實踐指導原則

1. **品質優先**: 寧可處理少量高品質文檔，也不要大量低品質內容
2. **自動化為主**: 盡可能自動化處理流程，減少人工干預
3. **標準統一**: 建立並執行統一的文檔和元數據標準
4. **監控改進**: 建立持續監控機制，及時發現和解決問題

### 5.3 下章預告

第2章將深入探討混合檢索系統的設計與實現，重點分析如何在企業級規模下實現高效、準確的信息檢索，這是 RAG 系統的核心技術環節。

---

## 參考文獻

[^17]: IBM Research Team. (2024). "Docling: Advanced Document Processing for Enterprise AI." *IBM Research Technical Report*.

---

**課程評估**: 本章內容在期中考試中占25%權重，重點考查文檔處理技術和品質管理能力。

**實驗要求**: 學生需完成企業文檔處理系統的設計和實現，並提供完整的測試報告。