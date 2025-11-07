# 文檔處理基礎：別讓垃圾數據毀了你的 RAG
## 第1章：把企業文檔變成可用的知識

**學習時間**: 2-3 小時
**前置知識**: 會用 Python，知道什麼是 PDF
**目標**: 學會處理企業文檔，別被垃圾數據坑死

---

## 🎯 核心問題

**企業文檔 = 災難現場**

大部分企業的文檔狀況：
- 📄 **格式混亂**: PDF/Word/PPT/Confluence 到處都是
- 🗓️ **版本混亂**: 2019年的文件還在用，沒人知道是不是最新版
- 🔍 **找不到**: 關鍵信息藏在某個深度目錄的 Excel 表格裡
- 🚫 **權限混亂**: 誰能看什麼，連 IT 都搞不清楚

**底線**: 垃圾進，垃圾出 (GIGO)。RAG 系統再聰明，也救不了爛數據。

---

## 🔧 解決方案：實用的文檔處理流水線

### 1.1 文檔處理的現實選擇

#### **工具選型：簡單有效**

```python
# 2025年實用組合
pip install docling              # IBM出品，PDF處理最強
pip install unstructured         # 備用方案，格式支援廣
pip install pypdf               # 簡單PDF，速度快
```

**選擇邏輯**:
- **主力**: Docling (準確率95%+，值得學習成本)
- **備用**: Unstructured (格式全，但準確率差點)
- **簡單**: PyPDF (純PDF場景，性能好)

#### **實際代碼：30行搞定基本處理**

```python
from docling.document_converter import DocumentConverter
from pathlib import Path

def process_enterprise_doc(file_path: str) -> dict:
    """處理企業文檔的最簡實現"""

    converter = DocumentConverter()

    try:
        # 轉換文檔
        result = converter.convert(file_path)

        # 提取純文本 (Markdown格式，保留結構)
        content = result.document.export_to_markdown()

        # 基本統計
        stats = {
            "char_count": len(content),
            "word_count": len(content.split()),
            "has_tables": "|" in content,
            "has_headers": "#" in content
        }

        return {
            "success": True,
            "content": content,
            "stats": stats,
            "file_path": file_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

# 批量處理
def process_document_folder(folder_path: str) -> list:
    """批量處理文檔文件夾"""

    results = []
    folder = Path(folder_path)

    # 支援的格式
    supported_formats = {'.pdf', '.docx', '.pptx', '.md', '.txt'}

    for file_path in folder.rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            result = process_enterprise_doc(str(file_path))
            results.append(result)

            # 簡單進度顯示
            status = "✅" if result["success"] else "❌"
            print(f"{status} {file_path.name}")

    return results
```

### 1.2 文檔分塊：別想太複雜

#### **分塊策略：實用主義**

學術界喜歡搞複雜的"語義分塊"。現實中，簡單的規則分塊就夠用：

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

def smart_chunk_document(content: str, doc_type: str = "general") -> list:
    """實用的文檔分塊策略"""

    # 不同類型文檔的分塊參數
    chunk_configs = {
        "technical": {"size": 800, "overlap": 100},   # 技術文檔要精確
        "policy": {"size": 1200, "overlap": 200},     # 政策文檔要完整
        "general": {"size": 1000, "overlap": 150}     # 一般文檔平衡
    }

    config = chunk_configs.get(doc_type, chunk_configs["general"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["size"],
        chunk_overlap=config["overlap"],
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " "]  # 中英文都考慮
    )

    chunks = splitter.split_text(content)

    # 添加基本元數據
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": f"chunk_{i}",
            "text": chunk,
            "char_count": len(chunk),
            "word_count": len(chunk.split()),
            "chunk_index": i
        })

    return chunk_data

# 測試效果
def test_chunking():
    """測試分塊效果"""

    sample_text = "你的測試文檔內容..."
    chunks = smart_chunk_document(sample_text, "technical")

    print(f"原文長度: {len(sample_text)}")
    print(f"分塊數量: {len(chunks)}")
    print(f"平均分塊長度: {sum(len(c['text']) for c in chunks) / len(chunks):.0f}")

    return chunks
```

#### **為什麼不用複雜的語義分塊？**

**Linus 觀點**:
> "複雜的算法通常是為了掩蓋設計問題。好的設計應該讓簡單的算法就能工作。"

**現實檢驗**:
- ✅ **簡單分塊** + **好的檢索** = 95% 場景夠用
- ❌ **複雜分塊** + **普通檢索** = 過度工程，性能還可能更差
- 🎯 **先做簡單版本，測量效果，確實不夠再優化**

---

## 💾 元數據：只要有用的

### 2.1 最小可行元數據

**別搞複雜的本體論！** 企業需要的元數據很簡單：

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import hashlib

@dataclass
class SimpleDocumentMetadata:
    """簡單實用的文檔元數據"""

    # 基本標識 (必需)
    doc_id: str
    title: str
    file_path: str
    content_hash: str  # 用於檢測變更

    # 分類信息 (重要)
    document_type: str  # "manual", "policy", "tech_spec", "general"
    department: str     # "engineering", "legal", "hr", "general"

    # 時間信息 (關鍵)
    created_at: datetime
    modified_at: datetime
    processed_at: datetime

    # 權限信息 (安全)
    access_level: str = "internal"  # "public", "internal", "confidential"
    owner: str = "unknown"

    # 內容統計 (有用)
    word_count: int = 0
    chunk_count: int = 0

    # 可選信息
    keywords: List[str] = None
    related_docs: List[str] = None

def extract_simple_metadata(file_path: str, content: str) -> SimpleDocumentMetadata:
    """提取簡單實用的元數據"""

    from pathlib import Path
    import os

    file_info = Path(file_path)

    # 從文件路徑推斷信息
    department = "general"
    doc_type = "general"

    # 簡單的路徑分析
    path_parts = str(file_info).lower().split('/')

    if any(dept in path_parts for dept in ["engineering", "tech", "dev"]):
        department = "engineering"
    elif any(dept in path_parts for dept in ["legal", "compliance"]):
        department = "legal"
    elif any(dept in path_parts for dept in ["hr", "people"]):
        department = "hr"

    if any(type_hint in path_parts for type_hint in ["manual", "guide"]):
        doc_type = "manual"
    elif any(type_hint in path_parts for type_hint in ["policy", "procedure"]):
        doc_type = "policy"
    elif any(type_hint in path_parts for type_hint in ["spec", "design"]):
        doc_type = "tech_spec"

    # 從文件名提取標題
    title = file_info.stem.replace('_', ' ').replace('-', ' ').title()

    # 時間信息
    try:
        file_stat = os.stat(file_path)
        created_at = datetime.fromtimestamp(file_stat.st_ctime)
        modified_at = datetime.fromtimestamp(file_stat.st_mtime)
    except:
        created_at = modified_at = datetime.now()

    return SimpleDocumentMetadata(
        doc_id=hashlib.md5(file_path.encode()).hexdigest()[:16],
        title=title,
        file_path=file_path,
        content_hash=hashlib.md5(content.encode()).hexdigest(),
        document_type=doc_type,
        department=department,
        created_at=created_at,
        modified_at=modified_at,
        processed_at=datetime.now(),
        word_count=len(content.split()),
        keywords=extract_simple_keywords(content)  # 簡單關鍵詞提取
    )

def extract_simple_keywords(content: str, max_keywords: int = 10) -> List[str]:
    """簡單的關鍵詞提取"""

    # 移除常用詞
    stopwords = {
        "的", "和", "在", "是", "有", "不", "了", "可以", "這個", "那個",
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"
    }

    # 簡單詞頻統計
    words = content.lower().split()
    word_freq = {}

    for word in words:
        if len(word) > 2 and word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1

    # 返回高頻詞
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in top_words[:max_keywords]]
```

### 2.2 文檔品質：只檢查重要的

**忘掉複雜的品質模型！** 實際上只需要檢查幾個關鍵點：

```python
def simple_quality_check(content: str, metadata: SimpleDocumentMetadata) -> dict:
    """簡單實用的品質檢查"""

    issues = []
    score = 1.0  # 從滿分開始扣分

    # 1. 內容長度檢查
    if len(content) < 100:
        issues.append("內容太短，可能是空文檔")
        score -= 0.3
    elif len(content) > 100000:
        issues.append("內容太長，可能需要拆分")
        score -= 0.1

    # 2. 亂碼檢查
    non_printable_ratio = sum(1 for c in content if not c.isprintable()) / len(content)
    if non_printable_ratio > 0.1:
        issues.append("可能包含亂碼或二進位數據")
        score -= 0.4

    # 3. 重複內容檢查
    lines = content.split('\n')
    unique_lines = set(line.strip() for line in lines if line.strip())
    if len(unique_lines) < len(lines) * 0.5:
        issues.append("重複內容過多")
        score -= 0.2

    # 4. 結構完整性檢查
    if metadata.document_type == "manual" and not any(word in content.lower()
                                                     for word in ["步驟", "操作", "step", "procedure"]):
        issues.append("手冊類文檔缺少操作步驟")
        score -= 0.2

    # 5. 時效性檢查 (超過2年的文檔要小心)
    doc_age_days = (datetime.now() - metadata.modified_at).days
    if doc_age_days > 730:  # 2年
        issues.append(f"文檔已有 {doc_age_days} 天未更新，可能過時")
        score -= min(0.3, (doc_age_days - 730) / 365 * 0.1)

    return {
        "quality_score": max(0.0, score),
        "grade": "A" if score >= 0.9 else "B" if score >= 0.7 else "C" if score >= 0.5 else "F",
        "issues": issues,
        "usable": score >= 0.5  # 低於50%就別用了
    }
```

---

## 📁 實際的文檔處理流水線

### 3.1 完整的處理流程

```python
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import time

class DocumentProcessor:
    """實用的文檔處理器"""

    def __init__(self):
        self.converter = DocumentConverter()

        # 簡單配置，別搞太複雜
        self.config = {
            "max_file_size_mb": 50,  # 50MB以上的文件別處理了
            "timeout_seconds": 60,   # 1分鐘處理不完就放棄
            "supported_formats": {".pdf", ".docx", ".pptx", ".md", ".txt"}
        }

    def process_folder(self, folder_path: str) -> Dict:
        """處理文檔文件夾"""

        print(f"🚀 開始處理文件夾: {folder_path}")
        start_time = time.time()

        # 找到所有支援的文檔
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in self.config["supported_formats"]:
                    # 檢查文件大小
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if size_mb <= self.config["max_file_size_mb"]:
                        all_files.append(file_path)
                    else:
                        print(f"⚠️  跳過大文件 ({size_mb:.1f}MB): {file_path}")

        print(f"📄 找到 {len(all_files)} 個可處理文件")

        # 並行處理 (但別開太多線程)
        max_workers = min(4, len(all_files))
        processed_docs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._process_single_file, all_files))

        # 統計結果
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        processing_time = time.time() - start_time

        print(f"✅ 處理完成: {len(successful)} 成功, {len(failed)} 失敗")
        print(f"⏱️  總耗時: {processing_time:.1f} 秒")

        return {
            "total_files": len(all_files),
            "successful": len(successful),
            "failed": len(failed),
            "processing_time": processing_time,
            "successful_docs": successful,
            "failed_docs": failed
        }

    def _process_single_file(self, file_path: str) -> Dict:
        """處理單個文件"""

        try:
            # 1. 轉換文檔
            result = process_enterprise_doc(file_path)

            if not result["success"]:
                return result

            content = result["content"]

            # 2. 提取元數據
            metadata = extract_simple_metadata(file_path, content)

            # 3. 品質檢查
            quality = simple_quality_check(content, metadata)

            # 4. 分塊處理
            chunks = smart_chunk_document(content, metadata.document_type)

            # 5. 組裝最終結果
            return {
                "success": True,
                "file_path": file_path,
                "metadata": metadata.__dict__,
                "quality": quality,
                "chunks": chunks,
                "usable": quality["usable"]
            }

        except Exception as e:
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e)
            }
```

---

## 🚨 PII 檢測：不能馬虎的安全檢查

### 4.1 實用的 PII 檢測

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def setup_simple_pii_detector():
    """設置簡單實用的 PII 檢測器"""

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # 企業常見的敏感信息類型
    pii_types = [
        "PERSON",           # 人名
        "EMAIL_ADDRESS",    # 郵箱
        "PHONE_NUMBER",     # 電話
        "CREDIT_CARD",      # 信用卡
        "US_SSN",          # 身份證號
        "IP_ADDRESS"        # IP地址
    ]

    return analyzer, anonymizer, pii_types

def check_document_pii(content: str) -> Dict:
    """檢查文檔中的個人信息"""

    analyzer, anonymizer, pii_types = setup_simple_pii_detector()

    # 檢測 PII
    results = analyzer.analyze(
        text=content,
        language="en",  # 主要支援英文，中文支援有限
        entities=pii_types
    )

    if not results:
        return {
            "has_pii": False,
            "risk_level": "safe",
            "detected_types": []
        }

    # 風險評級：簡單粗暴
    high_risk_types = {"CREDIT_CARD", "US_SSN"}
    detected_types = [r.entity_type for r in results]

    if any(pii_type in high_risk_types for pii_type in detected_types):
        risk_level = "high"
    elif len(detected_types) >= 3:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "has_pii": True,
        "risk_level": risk_level,
        "detected_types": detected_types,
        "detection_count": len(results),
        "needs_anonymization": risk_level in ["high", "medium"]
    }

def anonymize_if_needed(content: str, pii_check: Dict) -> str:
    """必要時進行匿名化"""

    if not pii_check["needs_anonymization"]:
        return content

    analyzer, anonymizer, _ = setup_simple_pii_detector()

    # 重新分析 (獲取位置信息)
    pii_results = analyzer.analyze(text=content, language="en")

    # 簡單替換策略
    anonymized_result = anonymizer.anonymize(
        text=content,
        analyzer_results=pii_results,
        operators={
            "PERSON": {"type": "replace", "new_value": "[PERSON]"},
            "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"},
            "PHONE_NUMBER": {"type": "replace", "new_value": "[PHONE]"},
            "CREDIT_CARD": {"type": "replace", "new_value": "[CARD]"},
            "US_SSN": {"type": "replace", "new_value": "[SSN]"}
        }
    )

    return anonymized_result.text
```

---

## 🏃‍♂️ 快速開始指南

### 5.1 30分鐘搭建文檔處理系統

```python
# main.py - 完整的文檔處理腳本
import sys
from pathlib import Path

def main():
    """主處理函數"""

    if len(sys.argv) < 2:
        print("用法: python main.py <文檔文件夾路徑>")
        return

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"❌ 文件夾不存在: {folder_path}")
        return

    # 1. 初始化處理器
    processor = DocumentProcessor()

    # 2. 批量處理
    results = processor.process_folder(folder_path)

    # 3. 過濾可用文檔
    usable_docs = [doc for doc in results["successful_docs"] if doc["usable"]]

    # 4. 保存結果 (JSON格式)
    output_file = f"processed_docs_{int(time.time())}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        import json
        json.dump({
            "processing_summary": {
                "total_files": results["total_files"],
                "successful": results["successful"],
                "usable": len(usable_docs),
                "processing_time": results["processing_time"]
            },
            "documents": usable_docs
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"📄 處理結果已保存到: {output_file}")
    print(f"📊 可用文檔數量: {len(usable_docs)}")

    # 5. 簡單統計
    if usable_docs:
        avg_chunks = sum(len(doc["chunks"]) for doc in usable_docs) / len(usable_docs)
        total_chunks = sum(len(doc["chunks"]) for doc in usable_docs)

        print(f"📈 總分塊數: {total_chunks}")
        print(f"📊 平均每文檔分塊數: {avg_chunks:.1f}")

        # 部門分佈
        dept_dist = {}
        for doc in usable_docs:
            dept = doc["metadata"]["department"]
            dept_dist[dept] = dept_dist.get(dept, 0) + 1

        print("🏢 部門分佈:")
        for dept, count in dept_dist.items():
            print(f"  {dept}: {count} 個文檔")

if __name__ == "__main__":
    main()
```

### 5.2 快速驗證腳本

```bash
# 測試處理效果
python main.py ./test_documents

# 檢查輸出
cat processed_docs_*.json | jq '.processing_summary'

# 檢查品質分佈
cat processed_docs_*.json | jq '.documents[].quality.grade' | sort | uniq -c
```

---

## 🎯 關鍵要點 (Linus Style)

### **做對的事情**

1. **先解決 80% 的問題**：簡單分塊 + 基本元數據就能解決大部分需求
2. **測量後優化**：別猜測性能瓶頸，用數據說話
3. **安全不能妥協**：PII 檢測和匿名化必須做對
4. **保持簡單**：複雜的設計通常是錯誤設計的徵象

### **避免的陷阱**

1. ❌ **過度工程**: 不要一開始就搞複雜的語義分析
2. ❌ **完美主義**: 不要試圖處理所有邊緣情況
3. ❌ **元數據膨脹**: 不要收集用不到的數據
4. ❌ **忽視性能**: 別讓處理時間超過用戶忍受範圍

### **成功檢查清單**

- ✅ 能處理企業常見格式 (PDF, Word, PPT)
- ✅ 分塊大小合理 (500-1500 字符)
- ✅ PII 檢測覆蓋主要類型
- ✅ 處理速度可接受 (秒級而非分鐘級)
- ✅ 錯誤處理健全 (單個文檔失敗不影響整個流程)

---

## 💡 實踐練習

### **練習 1: 文檔處理評估**
找一個真實的企業文檔集合，用我們的處理流程跑一遍：
- 統計處理成功率
- 分析品質分佈
- 檢查 PII 檢測效果

### **練習 2: 性能測試**
測試不同工具在你的環境下的性能：
- Docling vs PyPDF vs Unstructured
- 處理速度、準確率、資源使用

### **練習 3: 改進優化**
基於實際結果優化流程：
- 調整分塊參數
- 改進元數據提取
- 優化錯誤處理

---

## 🔧 下一步

第2章我們會學混合檢索，把這些處理好的文檔變成可檢索的向量。

**記住**: 文檔處理是基礎，做不好這一步，後面的再高級都沒用。

---

**實用提示**: 這章的代碼可以直接用在生產環境，先跑起來，有問題再優化。