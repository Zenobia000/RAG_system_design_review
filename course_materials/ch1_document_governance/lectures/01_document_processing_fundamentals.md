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
    """處理企業文檔 - 30秒學會版本"""

    converter = DocumentConverter()

    try:
        # 就這麼簡單：丟進去，拿結果
        result = converter.convert(file_path)
        content = result.document.export_to_markdown()

        # 別搞複雜統計，有用的就這幾個
        return {
            "success": True,
            "content": content,
            "word_count": len(content.split()),
            "looks_good": len(content) > 100,  # 太短通常是廢料
            "file_path": file_path
        }

    except Exception as e:
        # 失敗就失敗，別隱藏錯誤
        print(f"💥 處理失敗: {file_path} - {str(e)}")
        return {"success": False, "error": str(e), "file_path": file_path}

# 批量處理 - 簡單暴力有效
def process_document_folder(folder_path: str) -> dict:
    """批量處理文檔 - Linus風格：簡單粗暴有效"""

    from pathlib import Path
    import time

    print(f"🚀 開始處理: {folder_path}")
    start_time = time.time()

    # 找文件：支援常見格式就夠了
    supported = {'.pdf', '.docx', '.pptx', '.md', '.txt'}
    files = [f for f in Path(folder_path).rglob('*')
             if f.suffix.lower() in supported]

    print(f"📄 找到 {len(files)} 個文件")

    # 處理文件：別並行，簡單循環就好
    successful = []
    failed = []

    for file_path in files:
        result = process_enterprise_doc(str(file_path))

        if result["success"] and result["looks_good"]:
            successful.append(result)
            print(f"✅ {file_path.name}")
        else:
            failed.append(result)
            print(f"❌ {file_path.name}")

    elapsed = time.time() - start_time
    print(f"⏱️ 完成! {len(successful)}/{len(files)} 成功，耗時 {elapsed:.1f}秒")

    return {"successful": successful, "failed": failed, "stats": {
        "total": len(files), "success_rate": len(successful)/len(files)*100
    }}
```

### 1.2 文檔分塊：別想太複雜

#### **分塊策略：實用主義**

學術界喜歡搞複雜的"語義分塊"。現實中，簡單的規則分塊就夠用：

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

def chunk_document(content: str) -> list:
    """文檔分塊 - 一個配置搞定所有場景"""

    from langchain.text_splitters import RecursiveCharacterTextSplitter

    # 別搞複雜配置，一個參數組合應付80%場景
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # 1000字符，經驗最佳值
        chunk_overlap=200,      # 20%重疊，防止切斷關鍵信息
        separators=["\n\n", "\n", "。", ".", " "]
    )

    chunks = splitter.split_text(content)

    # 簡單包裝，別搞太多元數據
    return [{"text": chunk, "index": i} for i, chunk in enumerate(chunks)]

# 測試你的分塊效果
def test_chunking_quality(content: str) -> None:
    """快速測試分塊品質"""

    chunks = chunk_document(content)

    print(f"📊 分塊統計:")
    print(f"  原文: {len(content)} 字符")
    print(f"  分塊: {len(chunks)} 個")
    print(f"  平均: {len(content)//len(chunks)} 字符/塊")
    print(f"  最短: {min(len(c['text']) for c in chunks)}")
    print(f"  最長: {max(len(c['text']) for c in chunks)}")

    # 看看分塊邊界是否合理
    if len(chunks) > 1:
        print(f"📋 分塊示例:")
        print(f"  第1塊末尾: ...{chunks[0]['text'][-50:]}")
        print(f"  第2塊開頭: {chunks[1]['text'][:50]}...")

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

def extract_metadata(file_path: str, content: str) -> dict:
    """提取文檔元數據 - 實用版本，別搞複雜的類定義"""

    import os
    import hashlib
    from pathlib import Path

    # 基本信息：必須有的
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]  # 短點就夠
    title = Path(file_path).stem.replace('_', ' ').replace('-', ' ')

    # 從路徑猜測部門和類型 - 簡單粗暴但有效
    path_lower = file_path.lower()

    if any(x in path_lower for x in ['eng', 'tech', 'dev']):
        department = 'engineering'
    elif any(x in path_lower for x in ['legal', 'compliance']):
        department = 'legal'
    elif any(x in path_lower for x in ['hr', 'people']):
        department = 'hr'
    else:
        department = 'general'

    if any(x in path_lower for x in ['manual', 'guide', 'howto']):
        doc_type = 'manual'
    elif any(x in path_lower for x in ['policy', 'procedure', 'rule']):
        doc_type = 'policy'
    elif any(x in path_lower for x in ['spec', 'design', 'api']):
        doc_type = 'tech_spec'
    else:
        doc_type = 'general'

    # 時間信息
    try:
        stat = os.stat(file_path)
        modified = datetime.fromtimestamp(stat.st_mtime)
    except:
        modified = datetime.now()

    return {
        'id': doc_id,
        'title': title,
        'file_path': file_path,
        'department': department,
        'type': doc_type,
        'modified': modified,
        'word_count': len(content.split()),
        'is_old': (datetime.now() - modified).days > 365  # 超過1年算舊
    }

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
def quality_check(content: str, metadata: dict) -> dict:
    """文檔品質檢查 - 實用版本，只檢查會出事的問題"""

    issues = []

    # 1. 明顯的問題
    if len(content) < 50:
        return {"usable": False, "issue": "文檔太短，可能是空的"}

    # 2. 亂碼檢查 - 這個會搞壞 RAG
    weird_chars = sum(1 for c in content[:1000] if not c.isprintable() and c not in '\n\t')
    if weird_chars > 50:  # 前1000字符有50個以上奇怪字符
        return {"usable": False, "issue": "可能有亂碼"}

    # 3. 重複垃圾檢查
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if len(set(lines)) < len(lines) * 0.3:  # 70%以上重複行
        return {"usable": False, "issue": "重複內容太多"}

    # 4. 時效性警告
    if metadata.get('is_old', False):
        issues.append("文檔可能已過時")

    return {
        "usable": True,
        "issues": issues,
        "warning_count": len(issues)
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
    """文檔處理器 - Linus風格：簡單配置，專注核心功能"""

    def __init__(self):
        from docling.document_converter import DocumentConverter
        self.converter = DocumentConverter()

        # 配置：簡單明確，別搞一堆選項
        self.max_size_mb = 50     # 大文件直接跳過
        self.timeout = 60         # 60秒搞不定就算了
        self.formats = {".pdf", ".docx", ".pptx", ".md", ".txt"}

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

def check_for_sensitive_info(content: str) -> dict:
    """檢查敏感信息 - 簡化版，抓主要風險就夠了"""

    # 簡單正則表達式檢測常見敏感信息
    import re

    patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    }

    detected = {}
    for pii_type, pattern in patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            detected[pii_type] = len(matches)

    # 簡單風險評估
    if "credit_card" in detected or "ssn" in detected:
        risk = "high"
    elif len(detected) >= 2:
        risk = "medium"
    elif detected:
        risk = "low"
    else:
        risk = "safe"

    return {
        "has_sensitive_info": bool(detected),
        "risk_level": risk,
        "detected_types": list(detected.keys()),
        "total_matches": sum(detected.values()),
        "action": "anonymize" if risk in ["high", "medium"] else "proceed"
    }

def simple_anonymize(content: str, sensitive_check: dict) -> str:
    """簡單匿名化 - 直接替換，別搞複雜算法"""

    if not sensitive_check["has_sensitive_info"]:
        return content

    import re

    # 暴力替換法：簡單有效
    replacements = {
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[PHONE]',
        r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b': '[CARD]'
    }

    anonymized = content
    for pattern, replacement in replacements.items():
        anonymized = re.sub(pattern, replacement, anonymized)

    return anonymized

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
# 完整的文檔處理腳本 - 拿來就用
def process_all_documents(folder_path: str) -> str:
    """一個函數搞定所有文檔處理"""

    import json
    import time
    from pathlib import Path

    print(f"🚀 開始處理企業文檔: {folder_path}")

    # 1. 批量處理
    results = process_document_folder(folder_path)
    successful_docs = results["successful"]

    # 2. 處理每個成功的文檔
    final_docs = []
    for doc in successful_docs:
        # 提取元數據
        metadata = extract_metadata(doc["file_path"], doc["content"])

        # 品質檢查
        quality = quality_check(doc["content"], metadata)

        if quality["usable"]:
            # PII 檢查
            pii_check = check_for_sensitive_info(doc["content"])

            # 必要時匿名化
            clean_content = simple_anonymize(doc["content"], pii_check)

            # 分塊
            chunks = chunk_document(clean_content)

            final_docs.append({
                "metadata": metadata,
                "content": clean_content,
                "chunks": chunks,
                "quality": quality,
                "pii_info": pii_check
            })

    # 3. 保存結果
    output_file = f"enterprise_knowledge_base_{int(time.time())}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_processed": len(successful_docs),
                "usable_documents": len(final_docs),
                "total_chunks": sum(len(doc["chunks"]) for doc in final_docs),
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "documents": final_docs
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ 完成! 可用文檔: {len(final_docs)}")
    print(f"📄 結果保存在: {output_file}")

    return output_file

# 使用示例
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python doc_processor.py <文檔文件夾>")
        print("例子: python doc_processor.py ./company_docs")
        sys.exit(1)

    output_file = process_all_documents(sys.argv[1])
    print(f"🎉 知識庫準備完成: {output_file}")
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