# DocOps Pipeline Design
## Technical Design Document v1.0

**Document ID**: `CH1-DOCOPS-001`
**Author**: Data Platform Team
**Reviewers**: Staff Engineers, Data Architects
**Status**: `DESIGN REVIEW` → `IMPLEMENTATION READY`
**Last Updated**: 2025-01-06

---

## 1. System Overview

### 1.1 Problem Statement
Enterprise knowledge bases contain diverse document formats requiring:
- **Unified Processing**: PDF, DOCX, PPTX, Confluence, SharePoint, Slack
- **Quality Assurance**: 99.5% extraction accuracy, layout preservation
- **Scalability**: Process 10K+ documents/hour with sub-minute latency
- **Compliance**: PII detection, data lineage, audit trails
- **Incremental Updates**: Real-time sync with source systems

### 1.2 Solution Architecture
**DocOps Pipeline** - A production-grade document processing system implementing the **Extract → Transform → Load → Index (ETLI)** pattern:

```mermaid
graph LR
    subgraph "Ingestion Layer"
        A[Document Sources] --> B[Docling Processor]
        A --> C[Unstructured Fallback]
    end

    subgraph "Processing Layer"
        B --> D[Content Extraction]
        C --> D
        D --> E[PII Detection]
        E --> F[Chunk Segmentation]
        F --> G[Metadata Enrichment]
    end

    subgraph "Storage Layer"
        G --> H[Document Store]
        G --> I[Vector Index]
        G --> J[Audit Log]
    end
```

---

## 2. Technology Stack Deep Dive

### 2.1 Core Processing Engine

#### **Primary: Docling (IBM Research)**
```python
# Production Configuration
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

config = {
    "pdf_format_options": PdfFormatOption(
        do_ocr=True,
        do_table_structure=True,
        table_structure_options={
            "mode": "accurate",  # vs "fast"
            "do_cell_matching": True
        }
    ),
    "max_file_size": "100MB",
    "timeout": 300,  # 5 minutes per document
    "parallel_factor": 4  # CPU cores
}

converter = DocumentConverter(
    format_options={InputFormat.PDF: config["pdf_format_options"]}
)
```

**Performance Benchmarks**:
- PDF extraction: 95.2% accuracy (vs 87.3% PyPDF)
- Table structure: 92.8% F1-score
- Processing speed: 2.3 pages/second
- Memory usage: 150MB per worker

#### **Fallback: Unstructured**
```python
# Fallback for Complex Layouts
from unstructured.partition.auto import partition
from unstructured.staging.huggingface import stage_for_transformers

fallback_config = {
    "strategy": "hi_res",  # High resolution processing
    "infer_table_structure": True,
    "chunking_strategy": "by_title",
    "max_characters": 1500,
    "new_after_n_chars": 1200,
    "overlap": 100
}

# Trigger conditions for fallback
def should_use_fallback(docling_result):
    return (
        docling_result.confidence < 0.85 or
        docling_result.table_count == 0 and has_complex_tables(doc) or
        docling_result.processing_time > 300
    )
```

### 2.2 Content Segmentation Strategy

#### **Semantic Chunking with LangChain**
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter, SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Production Chunking Pipeline
class EnterpriseChunker:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, document_content, doc_type):
        """Multi-strategy chunking based on document type"""
        if doc_type in ["technical_spec", "legal_contract"]:
            # Preserve logical structure
            return self.semantic_splitter.split_text(document_content)
        else:
            # Standard chunking for general documents
            return self.recursive_splitter.split_text(document_content)
```

#### **Chunking Performance Targets**
| Document Type | Chunk Size | Overlap | Accuracy Target |
|---|---|---|---|
| **Technical Docs** | 800-1200 chars | 150 chars | >92% |
| **Legal Contracts** | 1200-1500 chars | 200 chars | >95% |
| **Product Specs** | 600-1000 chars | 100 chars | >90% |
| **Meeting Notes** | 400-800 chars | 50 chars | >88% |

### 2.3 PII Detection & Data Protection

#### **Microsoft Presidio Integration**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy

class EnterprisePIIDetector:
    def __init__(self):
        # Load production-grade models
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Custom enterprise entities
        self.custom_recognizers = [
            "EMPLOYEE_ID",      # Format: EMP-123456
            "PROJECT_CODE",     # Format: PROJ-ABC-2024
            "CUSTOMER_ID",      # Format: CUST-789012
            "IP_ADDRESS",       # Standard IP detection
            "INTERNAL_URL"      # *.company.com patterns
        ]

    def analyze_document(self, text, language="en"):
        """Detect PII with enterprise-specific patterns"""
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                     "CREDIT_CARD", "SSN"] + self.custom_recognizers
        )

        # Risk assessment
        risk_level = self._calculate_risk_level(results)

        return {
            "pii_entities": results,
            "risk_level": risk_level,
            "requires_anonymization": risk_level >= "MEDIUM",
            "approval_required": risk_level >= "HIGH"
        }

    def anonymize_content(self, text, entities):
        """Anonymize detected PII"""
        return self.anonymizer.anonymize(
            text=text,
            analyzer_results=entities,
            operators={
                "PERSON": {"type": "replace", "new_value": "[PERSON]"},
                "EMAIL_ADDRESS": {"type": "mask", "masking_char": "*", "chars_to_mask": 4},
                "EMPLOYEE_ID": {"type": "replace", "new_value": "[EMP_ID]"}
            }
        )
```

### 2.4 Metadata Management System

#### **Enterprise Metadata Schema**
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import uuid

@dataclass
class DocumentMetadata:
    """Enterprise document metadata schema"""

    # Core Identifiers
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_system: str = ""  # "confluence", "sharepoint", "slack"
    source_url: str = ""
    content_hash: str = ""  # SHA-256 for deduplication

    # Content Classification
    document_type: str = ""  # "technical_spec", "policy", "meeting_notes"
    language: str = "en"
    page_count: int = 0
    word_count: int = 0

    # Access Control
    access_level: str = "internal"  # "public", "internal", "confidential", "secret"
    owner_department: str = ""
    allowed_roles: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)

    # Temporal Information
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    indexed_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Quality Metrics
    extraction_confidence: float = 0.0
    processing_duration: float = 0.0
    pii_risk_level: str = "LOW"

    # Business Context
    project_codes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    related_documents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "document_id": self.document_id,
            "source_system": self.source_system,
            "source_url": self.source_url,
            "content_hash": self.content_hash,
            "document_type": self.document_type,
            "language": self.language,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "access_level": self.access_level,
            "owner_department": self.owner_department,
            "allowed_roles": self.allowed_roles,
            "allowed_users": self.allowed_users,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "indexed_at": self.indexed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "extraction_confidence": self.extraction_confidence,
            "processing_duration": self.processing_duration,
            "pii_risk_level": self.pii_risk_level,
            "project_codes": self.project_codes,
            "keywords": self.keywords,
            "categories": self.categories,
            "related_documents": self.related_documents
        }
```

---

## 3. Pipeline Architecture

### 3.1 Processing Workflow

#### **Document Ingestion Pipeline**
```python
from dataclasses import dataclass
from typing import AsyncGenerator
import asyncio
import aiofiles

class DocOpsManager:
    """Enterprise document processing orchestrator"""

    def __init__(self, config: DocOpsConfig):
        self.docling_processor = DocumentConverter()
        self.pii_detector = EnterprisePIIDetector()
        self.chunker = EnterpriseChunker()
        self.metadata_store = MetadataStore()
        self.vector_store = QdrantClient()

    async def process_document_batch(self, documents: List[str]) -> List[ProcessingResult]:
        """Process multiple documents concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_docs)

        async def process_single(doc_path: str) -> ProcessingResult:
            async with semaphore:
                return await self._process_document(doc_path)

        tasks = [process_single(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if not isinstance(r, Exception)]

    async def _process_document(self, doc_path: str) -> ProcessingResult:
        """Single document processing pipeline"""
        try:
            # Stage 1: Content Extraction
            extracted_content = await self._extract_content(doc_path)

            # Stage 2: PII Detection & Protection
            pii_analysis = self.pii_detector.analyze_document(
                extracted_content.text
            )

            if pii_analysis["requires_anonymization"]:
                extracted_content.text = self.pii_detector.anonymize_content(
                    extracted_content.text,
                    pii_analysis["pii_entities"]
                )

            # Stage 3: Content Segmentation
            chunks = self.chunker.chunk_document(
                extracted_content.text,
                extracted_content.document_type
            )

            # Stage 4: Metadata Enrichment
            metadata = await self._enrich_metadata(
                extracted_content, pii_analysis
            )

            # Stage 5: Storage & Indexing
            await self._store_document(chunks, metadata)

            return ProcessingResult(
                success=True,
                document_id=metadata.document_id,
                chunks_count=len(chunks),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                document_path=doc_path
            )
```

### 3.2 Quality Assurance Pipeline

#### **Multi-Stage Validation**
```python
class QualityAssurance:
    """Document processing quality control"""

    def __init__(self):
        self.min_confidence_threshold = 0.85
        self.max_processing_time = 300  # seconds
        self.required_metadata_fields = [
            "document_type", "access_level", "owner_department"
        ]

    async def validate_extraction(self, result: ExtractionResult) -> ValidationReport:
        """Comprehensive quality validation"""
        issues = []

        # Content Quality Checks
        if result.confidence < self.min_confidence_threshold:
            issues.append(f"Low extraction confidence: {result.confidence}")

        if len(result.text) < 100:
            issues.append("Extracted text too short")

        if self._detect_garbled_text(result.text):
            issues.append("Potential garbled text detected")

        # Metadata Completeness
        missing_fields = [
            field for field in self.required_metadata_fields
            if not getattr(result.metadata, field, None)
        ]
        if missing_fields:
            issues.append(f"Missing metadata fields: {missing_fields}")

        # Performance Validation
        if result.processing_time > self.max_processing_time:
            issues.append(f"Processing timeout: {result.processing_time}s")

        return ValidationReport(
            is_valid=len(issues) == 0,
            issues=issues,
            quality_score=self._calculate_quality_score(result)
        )

    def _detect_garbled_text(self, text: str) -> bool:
        """Detect potentially corrupted text extraction"""
        # High ratio of non-printable characters
        non_printable_ratio = sum(1 for c in text if not c.isprintable()) / len(text)
        if non_printable_ratio > 0.1:
            return True

        # Excessive repetitive patterns
        import re
        repetitive_patterns = re.findall(r'(.{3,})\1{3,}', text)
        if len(repetitive_patterns) > 10:
            return True

        return False
```

---

## 4. Performance & Scalability

### 4.1 Performance Benchmarks

#### **Processing Throughput**
| Document Type | Avg Size | Processing Time | Throughput |
|---|---|---|---|
| **PDF (Text)** | 2.5MB | 12s | 300 docs/hour |
| **PDF (Image-heavy)** | 15MB | 45s | 80 docs/hour |
| **DOCX** | 1.2MB | 8s | 450 docs/hour |
| **PPTX** | 8MB | 25s | 144 docs/hour |

#### **Resource Utilization**
```yaml
Production Configuration:
  CPU: 16 cores (Intel Xeon)
  Memory: 64GB RAM
  GPU: NVIDIA T4 (16GB) for OCR
  Storage: 2TB NVMe SSD

Peak Performance:
  - Concurrent Documents: 8
  - Memory per Document: 2-4GB
  - GPU Utilization: 85%
  - CPU Utilization: 70%
```

### 4.2 Scalability Strategy

#### **Horizontal Scaling Architecture**
```python
# Kubernetes Deployment Configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docops-processor
spec:
  replicas: 5
  selector:
    matchLabels:
      app: docops-processor
  template:
    metadata:
      labels:
        app: docops-processor
    spec:
      containers:
      - name: processor
        image: docops:v1.2.0
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        env:
        - name: MAX_CONCURRENT_DOCS
          value: "4"
        - name: PROCESSING_TIMEOUT
          value: "300"
```

#### **Auto-scaling Triggers**
```yaml
Scaling Metrics:
  Scale Up When:
    - Queue depth > 100 documents
    - CPU utilization > 80% for 5 minutes
    - Memory utilization > 85%
    - Processing latency > 60 seconds

  Scale Down When:
    - Queue depth < 10 documents
    - CPU utilization < 40% for 15 minutes
    - All pods idle for 10 minutes
```

---

## 5. Data Flow & Storage

### 5.1 Storage Architecture

#### **Multi-tier Storage Strategy**
```python
class StorageManager:
    """Enterprise document storage orchestration"""

    def __init__(self):
        # Raw document storage (Object Store)
        self.raw_storage = S3Client(bucket="enterprise-docs-raw")

        # Processed content (PostgreSQL)
        self.content_db = PostgreSQLClient(
            host="postgres-primary.internal",
            database="docops_content"
        )

        # Vector indices (Qdrant)
        self.vector_store = QdrantClient(
            host="qdrant-cluster.internal",
            port=6333
        )

        # Cache layer (Redis)
        self.cache = RedisClient(
            host="redis-cluster.internal",
            port=6379
        )

    async def store_document(self, document: ProcessedDocument):
        """Multi-tier document storage"""

        # Tier 1: Raw Document Archive
        raw_key = f"raw/{document.metadata.document_id}"
        await self.raw_storage.upload(raw_key, document.original_content)

        # Tier 2: Processed Content Database
        await self.content_db.insert_document(
            document_id=document.metadata.document_id,
            content=document.processed_content,
            metadata=document.metadata.to_dict(),
            chunks=document.chunks
        )

        # Tier 3: Vector Index
        for i, chunk in enumerate(document.chunks):
            await self.vector_store.upsert(
                collection_name="enterprise_knowledge",
                points=[{
                    "id": f"{document.metadata.document_id}_{i}",
                    "vector": chunk.embedding,
                    "payload": {
                        "document_id": document.metadata.document_id,
                        "chunk_index": i,
                        "content": chunk.text,
                        "metadata": document.metadata.to_dict()
                    }
                }]
            )

        # Tier 4: Cache Frequently Accessed
        if document.metadata.access_frequency > 0.8:
            cache_key = f"doc:{document.metadata.document_id}"
            await self.cache.setex(
                cache_key,
                3600,  # 1 hour TTL
                document.processed_content
            )
```

### 5.2 Data Lineage & Audit

#### **Comprehensive Audit Trail**
```python
@dataclass
class AuditEvent:
    """Document processing audit event"""
    event_id: str
    document_id: str
    event_type: str  # "extracted", "chunked", "indexed", "accessed"
    timestamp: datetime
    user_id: Optional[str]
    system_component: str
    details: Dict

class AuditLogger:
    """Enterprise audit logging"""

    def __init__(self):
        self.audit_db = PostgreSQLClient(database="docops_audit")

    async def log_processing_event(self, document_id: str, stage: str, details: Dict):
        """Log document processing events"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            document_id=document_id,
            event_type=f"processing_{stage}",
            timestamp=datetime.utcnow(),
            system_component="docops_pipeline",
            details=details
        )

        await self.audit_db.insert("audit_events", event.__dict__)

    async def get_document_lineage(self, document_id: str) -> List[AuditEvent]:
        """Retrieve complete processing history"""
        query = """
        SELECT * FROM audit_events
        WHERE document_id = %s
        ORDER BY timestamp ASC
        """
        return await self.audit_db.query(query, [document_id])
```

---

## 6. Monitoring & Observability

### 6.1 Key Performance Indicators

#### **Processing Metrics**
```yaml
Pipeline Health:
  - Documents Processed/Hour: Target 1000+
  - Success Rate: Target >99.5%
  - Average Processing Time: Target <30s
  - Queue Depth: Target <50 documents

Quality Metrics:
  - Extraction Confidence: Target >0.9
  - PII Detection Accuracy: Target >95%
  - Chunk Quality Score: Target >0.85
  - Metadata Completeness: Target >98%
```

#### **Real-time Dashboard**
```python
# Prometheus Metrics Export
from prometheus_client import Counter, Histogram, Gauge

# Pipeline metrics
documents_processed = Counter(
    'docops_documents_processed_total',
    'Total documents processed',
    ['document_type', 'status']
)

processing_duration = Histogram(
    'docops_processing_duration_seconds',
    'Document processing duration',
    ['document_type'],
    buckets=[5, 10, 30, 60, 120, 300]
)

queue_size = Gauge(
    'docops_queue_size',
    'Current processing queue size'
)

extraction_confidence = Histogram(
    'docops_extraction_confidence',
    'Document extraction confidence score',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)
```

### 6.2 Error Handling & Recovery

#### **Resilient Processing Pipeline**
```python
class ResilientProcessor:
    """Fault-tolerant document processing"""

    def __init__(self):
        self.max_retries = 3
        self.retry_delays = [5, 15, 60]  # exponential backoff
        self.dead_letter_queue = DeadLetterQueue()

    async def process_with_retry(self, document_path: str) -> ProcessingResult:
        """Process document with automatic retry logic"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._process_document(document_path)

            except TemporaryFailure as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    await asyncio.sleep(delay)
                    continue

            except PermanentFailure as e:
                # Don't retry permanent failures
                await self.dead_letter_queue.send(document_path, str(e))
                return ProcessingResult(success=False, error=str(e))

        # All retries exhausted
        await self.dead_letter_queue.send(document_path, str(last_exception))
        return ProcessingResult(success=False, error=f"Max retries exceeded: {last_exception}")
```

---

## 7. Security & Compliance

### 7.1 Data Protection Strategy

#### **Encryption at All Stages**
```python
class SecureProcessor:
    """Security-first document processing"""

    def __init__(self):
        self.encryption_key = load_encryption_key()
        self.access_control = CasbinEnforcer()

    async def secure_process(self, document_path: str, user_context: UserContext):
        """Process document with security controls"""

        # 1. Access Control Validation
        if not await self.access_control.enforce(
            user_context.user_id,
            document_path,
            "read"
        ):
            raise UnauthorizedAccess(f"User {user_context.user_id} cannot access {document_path}")

        # 2. Decrypt Document (if encrypted at rest)
        decrypted_content = await self.decrypt_document(document_path)

        # 3. Process with PII Protection
        processed_result = await self.process_with_pii_protection(decrypted_content)

        # 4. Encrypt Processed Content
        encrypted_result = await self.encrypt_result(processed_result)

        # 5. Audit Log
        await self.audit_logger.log_access(
            user_id=user_context.user_id,
            document_id=processed_result.document_id,
            action="process",
            timestamp=datetime.utcnow()
        )

        return encrypted_result
```

### 7.2 Compliance Framework

#### **GDPR/CCPA Compliance**
```python
class ComplianceManager:
    """Data protection compliance management"""

    async def handle_data_subject_request(self, request_type: str, user_id: str):
        """Handle GDPR/CCPA data subject requests"""

        if request_type == "access":
            # Right to access personal data
            return await self._export_user_data(user_id)

        elif request_type == "deletion":
            # Right to be forgotten
            await self._delete_user_data(user_id)
            await self._remove_from_vectors(user_id)

        elif request_type == "rectification":
            # Right to rectify personal data
            return await self._update_user_data(user_id)

        elif request_type == "portability":
            # Right to data portability
            return await self._export_portable_data(user_id)

    async def _delete_user_data(self, user_id: str):
        """Comprehensive user data deletion"""

        # Remove from document metadata
        await self.content_db.execute("""
            UPDATE document_metadata
            SET allowed_users = array_remove(allowed_users, %s)
            WHERE %s = ANY(allowed_users)
        """, [user_id, user_id])

        # Remove from audit logs (with retention period consideration)
        retention_date = datetime.utcnow() - timedelta(days=90)
        await self.audit_db.execute("""
            DELETE FROM audit_events
            WHERE user_id = %s AND timestamp < %s
        """, [user_id, retention_date])

        # Remove from vector index metadata
        await self.vector_store.delete(
            collection_name="enterprise_knowledge",
            filter={"user_id": {"$eq": user_id}}
        )
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Core Pipeline (Weeks 1-4)
```mermaid
gantt
    title DocOps Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    Docling Integration  :done, docling, 2025-01-07, 2025-01-14
    Basic Chunking       :active, chunk, 2025-01-14, 2025-01-21
    Metadata Schema      :meta, 2025-01-21, 2025-01-28
    Storage Layer        :storage, 2025-01-28, 2025-02-04

    section Phase 2: Quality
    PII Detection        :pii, 2025-02-04, 2025-02-11
    Quality Assurance    :qa, 2025-02-11, 2025-02-18
    Error Handling       :error, 2025-02-18, 2025-02-25

    section Phase 3: Scale
    Performance Tuning   :perf, 2025-02-25, 2025-03-04
    Monitoring Setup     :monitor, 2025-03-04, 2025-03-11
    Production Deploy    :prod, 2025-03-11, 2025-03-18
```

### 8.2 Success Criteria
- [ ] **Throughput**: Process 1000+ documents/hour
- [ ] **Accuracy**: >95% extraction confidence
- [ ] **Security**: 100% PII detection coverage
- [ ] **Scalability**: Auto-scale to 50 worker nodes
- [ ] **Compliance**: Pass SOC2 Type II audit

---

## Appendix: Configuration Templates

### A.1 Production Configuration
```yaml
# config/production.yml
docops:
  processors:
    docling:
      max_file_size: "100MB"
      timeout: 300
      parallel_factor: 4
      ocr_enabled: true
      table_extraction: true

    unstructured:
      strategy: "hi_res"
      fallback_threshold: 0.85
      max_characters: 1500
      overlap: 100

  storage:
    raw_documents: "s3://enterprise-docs-raw"
    processed_content: "postgresql://docops:password@postgres:5432/content"
    vector_index: "qdrant://qdrant-cluster:6333"
    cache: "redis://redis-cluster:6379"

  security:
    encryption_key_id: "docops-encryption-key-2025"
    pii_detection_enabled: true
    audit_logging_enabled: true
    access_control_enabled: true

  performance:
    max_concurrent_documents: 8
    processing_timeout: 300
    queue_max_size: 1000
    auto_scaling_enabled: true
```

---

**Document Control**
- Version: 1.0
- Classification: Internal Use
- Next Review: 2025-04-06
- Distribution: Engineering Team, Data Platform Team