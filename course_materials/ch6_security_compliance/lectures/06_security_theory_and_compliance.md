# 安全理論與企業合規框架
## 大學教科書 第6章：資訊安全與法規遵循的系統化設計

**課程編號**: CS785 - 企業級檢索增強生成系統
**章節**: 第6章 安全與合規
**學習時數**: 8小時
**先修課程**: 資訊安全基礎, 法規遵循, 第0-5章
**作者**: 資訊安全研究團隊 & 合規專家組
**最後更新**: 2025-01-06

---

## 📚 學習目標 (Learning Objectives)

完成本章學習後，學生應能夠:

1. **安全理論**: 掌握零信任架構和深度防禦的數學模型
2. **合規框架**: 理解 GDPR、SOC2、HIPAA 等法規的技術實現要求
3. **系統設計**: 設計符合企業安全標準的 RAG 系統架構
4. **風險管理**: 建立完整的安全風險評估和應對機制

---

## 1. 資訊安全的理論基礎

### 1.1 零信任架構的數學模型

#### **零信任原理的形式化定義**

**定義 1.1** (零信任模型): 零信任安全模型可以形式化為訪問控制函數：

$$\text{Access}(s, r, a) = \bigwedge_{i=1}^{n} \text{Policy}_i(s, r, a, \text{Context})$$

其中：
- $s$: 主體 (用戶、服務、設備)
- $r$: 資源 (數據、API、系統)
- $a$: 動作 (讀取、寫入、執行)
- $\text{Context}$: 環境上下文 (時間、地點、設備狀態)

**原則 1.1** (零信任基本原則):
1. **永不信任**: $\forall s, r, a: \text{Trust}(s) = \emptyset$
2. **始終驗證**: $\forall \text{Access}: \text{Verify}(\text{Identity}, \text{Context}, \text{Policy})$
3. **最小權限**: $\text{Privilege}(s, r) = \min(\text{Required}, \text{Granted})$

#### **深度防禦的層級模型**

**定理 1.1** (安全層級獨立性): 在理想的深度防禦系統中，各安全層級應滿足獨立性條件：

$$P(\text{Breach}_{i+1} | \text{Breach}_i) < P(\text{Breach}_{i+1})$$

即上層被突破不應增加下層被攻破的概率。

**安全層級定義**:

```python
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import hashlib
import jwt
from datetime import datetime, timedelta

class SecurityLayer(Enum):
    """安全防禦層級"""
    PERIMETER = "perimeter"          # 周邊安全 (防火牆、WAF)
    IDENTITY = "identity"            # 身份認證 (MFA、SSO)
    ACCESS = "access"                # 訪問控制 (RBAC、ABAC)
    APPLICATION = "application"      # 應用安全 (輸入驗證、輸出過濾)
    DATA = "data"                    # 數據安全 (加密、DLP)
    MONITORING = "monitoring"        # 安全監控 (SIEM、審計)

@dataclass
class SecurityPolicy:
    """安全政策數據結構"""
    policy_id: str
    name: str
    description: str
    layer: SecurityLayer
    rules: List[Dict]
    enforcement_level: str  # advisory, warning, blocking
    applicable_resources: List[str]
    exceptions: List[Dict]

class ZeroTrustSecurityFramework:
    """零信任安全框架"""

    def __init__(self):
        self.security_layers = {layer: [] for layer in SecurityLayer}
        self.policy_engine = PolicyEngine()
        self.context_analyzer = ContextAnalyzer()
        self.risk_calculator = RiskCalculator()

    async def evaluate_access_request(self, request: Dict) -> Dict:
        """評估訪問請求"""

        # 1. 身份驗證
        identity_verification = await self._verify_identity(request["subject"])

        if not identity_verification["verified"]:
            return {
                "access_granted": False,
                "reason": "Identity verification failed",
                "verification_details": identity_verification
            }

        # 2. 上下文分析
        context_analysis = await self.context_analyzer.analyze_request_context(request)

        # 3. 風險評估
        risk_assessment = await self.risk_calculator.calculate_access_risk(
            request, identity_verification, context_analysis
        )

        # 4. 政策評估
        policy_evaluation = await self._evaluate_all_policies(
            request, context_analysis, risk_assessment
        )

        # 5. 最終決策
        access_decision = await self._make_access_decision(
            identity_verification, risk_assessment, policy_evaluation
        )

        return {
            "access_granted": access_decision["granted"],
            "reason": access_decision["reason"],
            "identity_verification": identity_verification,
            "context_analysis": context_analysis,
            "risk_assessment": risk_assessment,
            "policy_evaluation": policy_evaluation,
            "session_token": access_decision.get("session_token"),
            "access_duration": access_decision.get("access_duration")
        }

    async def _verify_identity(self, subject: Dict) -> Dict:
        """多因子身份驗證"""

        verification_factors = []

        # 第一因子: 密碼或證書
        primary_auth = await self._verify_primary_credential(subject)
        verification_factors.append(("primary", primary_auth))

        # 第二因子: MFA (如果需要)
        if self._requires_mfa(subject):
            mfa_result = await self._verify_mfa(subject)
            verification_factors.append(("mfa", mfa_result))

        # 第三因子: 設備信任 (如果配置)
        if self._requires_device_verification(subject):
            device_verification = await self._verify_device_trust(subject)
            verification_factors.append(("device", device_verification))

        # 綜合驗證結果
        all_factors_passed = all(result["verified"] for _, result in verification_factors)

        verification_strength = sum(
            result["confidence"] for _, result in verification_factors
        ) / len(verification_factors)

        return {
            "verified": all_factors_passed,
            "verification_factors": dict(verification_factors),
            "verification_strength": verification_strength,
            "multi_factor_used": len(verification_factors) > 1
        }

    async def _evaluate_all_policies(self, request: Dict,
                                   context: Dict,
                                   risk: Dict) -> Dict:
        """評估所有適用的安全政策"""

        applicable_policies = await self._find_applicable_policies(request)

        policy_results = {}
        overall_compliance = True

        for policy in applicable_policies:
            policy_result = await self.policy_engine.evaluate_policy(
                policy, request, context, risk
            )

            policy_results[policy.policy_id] = policy_result

            if policy_result["enforcement_level"] == "blocking" and not policy_result["compliant"]:
                overall_compliance = False

        return {
            "overall_compliance": overall_compliance,
            "policy_results": policy_results,
            "total_policies_evaluated": len(applicable_policies),
            "blocking_violations": [
                policy_id for policy_id, result in policy_results.items()
                if result["enforcement_level"] == "blocking" and not result["compliant"]
            ]
        }

    async def _make_access_decision(self, identity: Dict, risk: Dict, policies: Dict) -> Dict:
        """做出最終訪問決策"""

        # 決策邏輯
        if not identity["verified"]:
            return {"granted": False, "reason": "Identity verification failed"}

        if not policies["overall_compliance"]:
            return {
                "granted": False,
                "reason": "Policy violations detected",
                "violations": policies["blocking_violations"]
            }

        if risk["risk_level"] == "critical":
            return {"granted": False, "reason": "Risk level too high"}

        # 計算訪問權限級別
        access_level = self._calculate_access_level(identity, risk, policies)

        # 生成會話令牌
        session_token = await self._generate_session_token(identity, access_level)

        # 確定訪問期限
        access_duration = self._calculate_access_duration(risk["risk_level"], access_level)

        return {
            "granted": True,
            "reason": "All security checks passed",
            "access_level": access_level,
            "session_token": session_token,
            "access_duration": access_duration
        }
```

---

## 2. 個人資料保護與隱私工程

### 2.1 PII 檢測的理論基礎

#### **隱私敏感度的數學模型**

**定義 2.1** (隱私敏感度): 對於資料元素 $d$，其隱私敏感度定義為：

$$\text{Sensitivity}(d) = \alpha \cdot \text{Identifiability}(d) + \beta \cdot \text{Linkability}(d) + \gamma \cdot \text{Inference}(d)$$

其中：
- $\text{Identifiability}(d)$: 直接識別個人的能力
- $\text{Linkability}(d)$: 與其他數據關聯的能力
- $\text{Inference}(d)$: 推斷額外資訊的能力

**定理 2.1** (k-匿名性): 資料集 $D$ 滿足 k-匿名性當且僅當：

$$\forall d \in D: |\{d' \in D : \text{QI}(d) = \text{QI}(d')\}| \geq k$$

其中 $\text{QI}(d)$ 為準識別符集合。

#### **企業級 PII 檢測系統**

```python
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
import spacy
from typing import Dict, List, Any, Optional
import re

class EnterprisePIIDetector:
    """企業級 PII 檢測系統"""

    def __init__(self):
        # 初始化 Presidio 分析器
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # 添加企業特定實體識別器
        self._add_enterprise_recognizers()

        # 風險分級配置
        self.risk_levels = {
            "CRITICAL": ["SSN", "CREDIT_CARD", "BANK_ACCOUNT", "PASSPORT"],
            "HIGH": ["PHONE_NUMBER", "EMAIL_ADDRESS", "EMPLOYEE_ID", "MEDICAL_RECORD"],
            "MEDIUM": ["PERSON", "IP_ADDRESS", "LOCATION"],
            "LOW": ["ORGANIZATION", "DATE_TIME"]
        }

    def _add_enterprise_recognizers(self):
        """添加企業特定的 PII 識別器"""

        # 員工 ID 識別器
        employee_id_recognizer = PatternRecognizer(
            supported_entity="EMPLOYEE_ID",
            patterns=[{
                "name": "employee_id_pattern",
                "regex": r"\b(?:EMP|EMPL|E)-?\d{6,8}\b",
                "score": 0.85
            }]
        )

        # 客戶 ID 識別器
        customer_id_recognizer = PatternRecognizer(
            supported_entity="CUSTOMER_ID",
            patterns=[{
                "name": "customer_id_pattern",
                "regex": r"\b(?:CUST|CST|C)-?\d{8,12}\b",
                "score": 0.85
            }]
        )

        # 項目代碼識別器
        project_code_recognizer = PatternRecognizer(
            supported_entity="PROJECT_CODE",
            patterns=[{
                "name": "project_code_pattern",
                "regex": r"\b(?:PROJ|PRJ)-[A-Z]{2,4}-\d{4}\b",
                "score": 0.90
            }]
        )

        # 內部 URL 識別器
        internal_url_recognizer = PatternRecognizer(
            supported_entity="INTERNAL_URL",
            patterns=[{
                "name": "internal_url_pattern",
                "regex": r"https?://[\w\-\.]+\.(?:company\.com|internal\.net)[/\w\-\.]*",
                "score": 0.95
            }]
        )

        # 註冊識別器
        recognizers = [
            employee_id_recognizer,
            customer_id_recognizer,
            project_code_recognizer,
            internal_url_recognizer
        ]

        for recognizer in recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

    async def comprehensive_pii_analysis(self, text: str,
                                       context: Dict = None) -> Dict:
        """全面 PII 分析"""

        # 1. 基礎 PII 檢測
        analyzer_results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self._get_detection_entities(context),
            return_decision_process=True
        )

        # 2. 風險等級評估
        risk_assessment = await self._assess_pii_risk(analyzer_results, context)

        # 3. 合規要求分析
        compliance_requirements = await self._analyze_compliance_requirements(
            analyzer_results, context
        )

        # 4. 匿名化建議
        anonymization_plan = await self._create_anonymization_plan(
            analyzer_results, risk_assessment, compliance_requirements
        )

        return {
            "detected_entities": self._format_detection_results(analyzer_results),
            "risk_assessment": risk_assessment,
            "compliance_requirements": compliance_requirements,
            "anonymization_plan": anonymization_plan,
            "privacy_score": self._calculate_privacy_score(risk_assessment)
        }

    async def _assess_pii_risk(self, analyzer_results: List,
                             context: Dict = None) -> Dict:
        """評估 PII 風險等級"""

        risk_factors = {
            "entity_types": [],
            "entity_count": len(analyzer_results),
            "high_risk_count": 0,
            "cross_reference_potential": 0.0
        }

        # 統計不同類型的實體
        entity_type_counts = {}
        for result in analyzer_results:
            entity_type = result.entity_type
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

            # 計算風險等級
            for risk_level, entity_types in self.risk_levels.items():
                if entity_type in entity_types:
                    risk_factors["entity_types"].append((entity_type, risk_level))
                    if risk_level in ["CRITICAL", "HIGH"]:
                        risk_factors["high_risk_count"] += 1

        # 評估交叉引用風險
        if len(entity_type_counts) > 1:
            # 多種類型的 PII 存在交叉引用風險
            risk_factors["cross_reference_potential"] = min(1.0, len(entity_type_counts) / 5.0)

        # 計算總體風險等級
        if risk_factors["high_risk_count"] > 0:
            overall_risk = "HIGH"
        elif risk_factors["entity_count"] >= 5:
            overall_risk = "MEDIUM"
        elif risk_factors["entity_count"] >= 1:
            overall_risk = "LOW"
        else:
            overall_risk = "NONE"

        return {
            "overall_risk_level": overall_risk,
            "risk_factors": risk_factors,
            "entity_distribution": entity_type_counts,
            "requires_anonymization": overall_risk in ["HIGH", "MEDIUM"],
            "requires_approval": overall_risk == "HIGH"
        }

    async def _analyze_compliance_requirements(self, analyzer_results: List,
                                             context: Dict = None) -> Dict:
        """分析合規要求"""

        compliance_frameworks = {}

        detected_entity_types = set(result.entity_type for result in analyzer_results)

        # GDPR 分析
        gdpr_entities = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"}
        if detected_entity_types & gdpr_entities:
            compliance_frameworks["GDPR"] = {
                "applicable": True,
                "triggered_by": list(detected_entity_types & gdpr_entities),
                "requirements": [
                    "數據處理法律基礎",
                    "數據主體權利實施",
                    "數據保護影響評估",
                    "同意管理機制"
                ]
            }

        # HIPAA 分析
        hipaa_entities = {"MEDICAL_RECORD", "PATIENT_ID", "HEALTH_INFO"}
        if detected_entity_types & hipaa_entities or context.get("domain") == "healthcare":
            compliance_frameworks["HIPAA"] = {
                "applicable": True,
                "triggered_by": list(detected_entity_types & hipaa_entities),
                "requirements": [
                    "業務夥伴協議 (BAA)",
                    "最小必要原則",
                    "加密要求",
                    "審計追蹤"
                ]
            }

        # PCI DSS 分析
        pci_entities = {"CREDIT_CARD", "BANK_ACCOUNT"}
        if detected_entity_types & pci_entities:
            compliance_frameworks["PCI_DSS"] = {
                "applicable": True,
                "triggered_by": list(detected_entity_types & pci_entities),
                "requirements": [
                    "數據加密",
                    "網絡分割",
                    "訪問控制",
                    "定期安全測試"
                ]
            }

        return compliance_frameworks

    async def intelligent_anonymization(self, text: str,
                                      pii_analysis: Dict,
                                      anonymization_strategy: str = "adaptive") -> Dict:
        """智能匿名化處理"""

        detected_entities = pii_analysis["detected_entities"]
        risk_level = pii_analysis["risk_assessment"]["overall_risk_level"]

        # 根據風險等級選擇匿名化策略
        if anonymization_strategy == "adaptive":
            if risk_level == "HIGH":
                strategy = "redaction"     # 完全遮蔽
            elif risk_level == "MEDIUM":
                strategy = "replacement"   # 替換為類型標籤
            else:
                strategy = "masking"       # 部分遮蔽
        else:
            strategy = anonymization_strategy

        # 配置匿名化操作
        anonymization_operators = self._configure_anonymization_operators(
            detected_entities, strategy
        )

        # 執行匿名化
        anonymization_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=detected_entities,
            operators=anonymization_operators
        )

        # 驗證匿名化效果
        post_anonymization_check = await self._verify_anonymization_completeness(
            anonymization_result.text, detected_entities
        )

        return {
            "anonymized_text": anonymization_result.text,
            "strategy_used": strategy,
            "entities_processed": len(detected_entities),
            "anonymization_items": [
                {
                    "entity_type": item.entity_type,
                    "original_text": item.text,
                    "anonymized_text": item.anonymized_text,
                    "operator": item.operator
                }
                for item in anonymization_result.items
            ],
            "completeness_verification": post_anonymization_check
        }

    def _configure_anonymization_operators(self, entities: List,
                                         strategy: str) -> Dict[str, OperatorConfig]:
        """配置匿名化操作器"""

        operators = {}

        for entity in entities:
            entity_type = entity.entity_type

            if strategy == "redaction":
                operators[entity_type] = OperatorConfig("redact", {"new_value": "[REDACTED]"})

            elif strategy == "replacement":
                replacement_values = {
                    "PERSON": "[PERSON]",
                    "EMAIL_ADDRESS": "[EMAIL]",
                    "PHONE_NUMBER": "[PHONE]",
                    "EMPLOYEE_ID": "[EMP_ID]",
                    "CUSTOMER_ID": "[CUSTOMER_ID]",
                    "CREDIT_CARD": "[CREDIT_CARD]",
                    "SSN": "[SSN]",
                    "IP_ADDRESS": "[IP_ADDRESS]"
                }

                replacement_value = replacement_values.get(entity_type, f"[{entity_type}]")
                operators[entity_type] = OperatorConfig("replace", {"new_value": replacement_value})

            elif strategy == "masking":
                if entity_type in ["EMAIL_ADDRESS", "PHONE_NUMBER"]:
                    operators[entity_type] = OperatorConfig("mask", {
                        "masking_char": "*",
                        "chars_to_mask": 4,
                        "from_end": False
                    })
                elif entity_type == "CREDIT_CARD":
                    operators[entity_type] = OperatorConfig("mask", {
                        "masking_char": "*",
                        "chars_to_mask": 12,
                        "from_end": False
                    })
                else:
                    operators[entity_type] = OperatorConfig("replace", {"new_value": f"[{entity_type}]"})

        return operators
```

---

## 3. 企業合規自動化

### 3.1 GDPR 合規的技術實現

#### **數據主體權利的系統化實現**

**權利 3.1** (GDPR 數據主體權利的技術映射):

| GDPR 權利 | 技術實現 | 系統組件 |
|-----------|---------|---------|
| **訪問權** (Art. 15) | 數據導出 API | 用戶數據查詢系統 |
| **更正權** (Art. 16) | 數據修改 API | 向量索引更新機制 |
| **刪除權** (Art. 17) | 數據刪除 API | 分散式數據清理 |
| **可攜性權** (Art. 20) | 標準格式導出 | 數據序列化系統 |

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json

@dataclass
class DataSubjectRequest:
    """數據主體請求"""
    request_id: str
    request_type: str  # access, rectification, erasure, portability
    data_subject_id: str
    request_details: Dict
    submitted_at: datetime
    status: str
    processed_by: Optional[str]

class GDPRComplianceManager:
    """GDPR 合規管理器"""

    def __init__(self):
        self.data_inventory = DataInventoryManager()
        self.consent_manager = ConsentManager()
        self.audit_logger = AuditLogger()
        self.notification_system = NotificationSystem()

    async def handle_data_subject_request(self, request: DataSubjectRequest) -> Dict:
        """處理數據主體請求"""

        # 1. 身份驗證
        identity_verification = await self._verify_data_subject_identity(
            request.data_subject_id, request.request_details
        )

        if not identity_verification["verified"]:
            return {
                "status": "rejected",
                "reason": "身份驗證失敗",
                "verification_details": identity_verification
            }

        # 2. 請求有效性檢查
        validity_check = await self._validate_request(request)

        if not validity_check["valid"]:
            return {
                "status": "rejected",
                "reason": "請求無效",
                "validation_details": validity_check
            }

        # 3. 數據範圍確定
        data_scope = await self._determine_data_scope(request)

        # 4. 執行數據主體權利
        execution_result = await self._execute_data_subject_right(request, data_scope)

        # 5. 審計記錄
        await self.audit_logger.log_data_subject_request(
            request, execution_result, identity_verification
        )

        # 6. 通知相關方
        await self._notify_stakeholders(request, execution_result)

        return {
            "status": "completed",
            "request_id": request.request_id,
            "execution_result": execution_result,
            "completion_time": datetime.now(),
            "audit_trail_id": execution_result.get("audit_trail_id")
        }

    async def _execute_data_subject_right(self, request: DataSubjectRequest,
                                        data_scope: Dict) -> Dict:
        """執行數據主體權利"""

        execution_results = {}

        if request.request_type == "access":
            # 數據訪問權實現
            access_result = await self._execute_access_right(request, data_scope)
            execution_results["access_result"] = access_result

        elif request.request_type == "rectification":
            # 數據更正權實現
            rectification_result = await self._execute_rectification_right(request, data_scope)
            execution_results["rectification_result"] = rectification_result

        elif request.request_type == "erasure":
            # 數據刪除權實現 (被遺忘權)
            erasure_result = await self._execute_erasure_right(request, data_scope)
            execution_results["erasure_result"] = erasure_result

        elif request.request_type == "portability":
            # 數據可攜性權實現
            portability_result = await self._execute_portability_right(request, data_scope)
            execution_results["portability_result"] = portability_result

        return execution_results

    async def _execute_erasure_right(self, request: DataSubjectRequest,
                                   data_scope: Dict) -> Dict:
        """執行數據刪除權 (技術實現)"""

        data_subject_id = request.data_subject_id
        erasure_results = {}

        # 1. 文檔內容中的個人數據刪除
        document_erasure = await self._erase_from_documents(
            data_subject_id, data_scope["documents"]
        )
        erasure_results["documents"] = document_erasure

        # 2. 向量索引中的數據刪除
        vector_erasure = await self._erase_from_vector_index(
            data_subject_id, data_scope["vector_data"]
        )
        erasure_results["vector_index"] = vector_erasure

        # 3. 元數據和日誌中的數據刪除
        metadata_erasure = await self._erase_from_metadata(
            data_subject_id, data_scope["metadata"]
        )
        erasure_results["metadata"] = metadata_erasure

        # 4. 審計日誌的特殊處理 (法律要求保留)
        audit_processing = await self._process_audit_logs_for_erasure(
            data_subject_id, data_scope["audit_logs"]
        )
        erasure_results["audit_logs"] = audit_processing

        # 5. 第三方系統通知
        third_party_notifications = await self._notify_third_party_processors(
            data_subject_id, request
        )
        erasure_results["third_party_notifications"] = third_party_notifications

        # 6. 驗證刪除完整性
        verification_result = await self._verify_erasure_completeness(
            data_subject_id, erasure_results
        )

        return {
            "erasure_results": erasure_results,
            "verification": verification_result,
            "completeness_score": verification_result["completeness_percentage"],
            "estimated_impact": self._estimate_erasure_impact(erasure_results)
        }

    async def _erase_from_vector_index(self, data_subject_id: str,
                                     vector_data_scope: Dict) -> Dict:
        """從向量索引中刪除數據"""

        erasure_stats = {
            "vectors_examined": 0,
            "vectors_deleted": 0,
            "collections_affected": [],
            "indexes_rebuilt": []
        }

        for collection_name, vector_ids in vector_data_scope.items():
            try:
                # 查詢包含個人數據的向量
                affected_vectors = await self._identify_personal_vectors(
                    collection_name, data_subject_id
                )

                erasure_stats["vectors_examined"] += len(affected_vectors)

                # 刪除向量
                if affected_vectors:
                    deletion_result = await self._delete_vectors_from_collection(
                        collection_name, affected_vectors
                    )

                    erasure_stats["vectors_deleted"] += deletion_result["deleted_count"]
                    erasure_stats["collections_affected"].append(collection_name)

                    # 如果刪除量大，重建索引以優化性能
                    if deletion_result["deleted_count"] > 1000:
                        rebuild_result = await self._rebuild_vector_index(collection_name)
                        if rebuild_result["success"]:
                            erasure_stats["indexes_rebuilt"].append(collection_name)

            except Exception as e:
                erasure_stats[f"error_{collection_name}"] = str(e)

        return erasure_stats

    async def _verify_erasure_completeness(self, data_subject_id: str,
                                         erasure_results: Dict) -> Dict:
        """驗證刪除完整性"""

        verification_checks = {}

        # 1. 文檔搜索驗證
        doc_search_result = await self._search_for_personal_data_in_documents(data_subject_id)
        verification_checks["document_search"] = {
            "data_found": len(doc_search_result) > 0,
            "found_locations": doc_search_result
        }

        # 2. 向量索引搜索驗證
        vector_search_result = await self._search_for_personal_data_in_vectors(data_subject_id)
        verification_checks["vector_search"] = {
            "data_found": len(vector_search_result) > 0,
            "found_vectors": vector_search_result
        }

        # 3. 元數據檢查
        metadata_check = await self._check_metadata_for_personal_data(data_subject_id)
        verification_checks["metadata_check"] = metadata_check

        # 計算完整性百分比
        total_checks = len(verification_checks)
        passed_checks = sum(1 for check in verification_checks.values()
                          if not check.get("data_found", True))

        completeness_percentage = (passed_checks / total_checks) * 100

        return {
            "verification_checks": verification_checks,
            "completeness_percentage": completeness_percentage,
            "fully_compliant": completeness_percentage == 100,
            "remaining_data_locations": self._identify_remaining_data(verification_checks)
        }

class SOC2ComplianceFramework:
    """SOC2 合規框架"""

    def __init__(self):
        self.trust_service_criteria = {
            "security": SecurityControlFramework(),
            "availability": AvailabilityControlFramework(),
            "processing_integrity": ProcessingIntegrityFramework(),
            "confidentiality": ConfidentialityFramework(),
            "privacy": PrivacyFramework()
        }

    async def assess_soc2_compliance(self, system_config: Dict) -> Dict:
        """評估 SOC2 合規狀況"""

        compliance_assessment = {}

        for criterion, framework in self.trust_service_criteria.items():
            criterion_assessment = await framework.assess_compliance(system_config)
            compliance_assessment[criterion] = criterion_assessment

        # 計算總體合規分數
        overall_score = sum(
            assessment["compliance_score"]
            for assessment in compliance_assessment.values()
        ) / len(compliance_assessment)

        # 識別合規差距
        compliance_gaps = []
        for criterion, assessment in compliance_assessment.items():
            for control in assessment["control_assessments"]:
                if not control["implemented"]:
                    compliance_gaps.append({
                        "criterion": criterion,
                        "control": control["control_id"],
                        "description": control["description"],
                        "priority": control["priority"],
                        "implementation_effort": control["estimated_effort"]
                    })

        return {
            "overall_compliance_score": overall_score,
            "criterion_assessments": compliance_assessment,
            "compliance_gaps": compliance_gaps,
            "readiness_level": self._classify_soc2_readiness(overall_score),
            "remediation_plan": self._create_soc2_remediation_plan(compliance_gaps)
        }
```

---

## 4. 安全監控與事件回應

### 4.1 安全事件檢測理論

#### **異常檢測的統計模型**

**定義 4.1** (安全異常): 給定正常行為模式的概率分佈 $P(\text{Normal})$，安全異常定義為：

$$\text{Anomaly} = \{x : P(x | \text{Normal}) < \tau\}$$

其中 $\tau$ 為異常檢測閾值。

**算法 4.1** (基於機器學習的異常檢測):

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class SecurityAnomalyDetector:
    """安全異常檢測器"""

    def __init__(self):
        # 異常檢測模型
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # 預期異常比例
            random_state=42,
            n_estimators=100
        )

        # 特徵標準化器
        self.scaler = StandardScaler()

        # 行為基線
        self.behavior_baselines = {}

        # 異常類型分類器
        self.anomaly_classifier = AnomalyTypeClassifier()

    async def train_anomaly_detector(self, training_data: List[Dict]) -> Dict:
        """訓練異常檢測模型"""

        # 1. 特徵工程
        feature_matrix = await self._extract_security_features(training_data)

        # 2. 數據預處理
        scaled_features = self.scaler.fit_transform(feature_matrix)

        # 3. 模型訓練
        self.isolation_forest.fit(scaled_features)

        # 4. 建立行為基線
        self.behavior_baselines = await self._establish_behavior_baselines(training_data)

        # 5. 模型驗證
        validation_result = await self._validate_anomaly_model(training_data)

        return {
            "training_samples": len(training_data),
            "feature_dimensions": feature_matrix.shape[1],
            "model_performance": validation_result,
            "baseline_establishment": "completed"
        }

    async def detect_security_anomalies(self, current_activity: List[Dict]) -> Dict:
        """檢測安全異常"""

        if not current_activity:
            return {"anomalies": [], "normal_activities": 0}

        # 1. 特徵提取
        activity_features = await self._extract_security_features(current_activity)

        # 2. 特徵標準化
        scaled_features = self.scaler.transform(activity_features)

        # 3. 異常檢測
        anomaly_scores = self.isolation_forest.decision_function(scaled_features)
        anomaly_predictions = self.isolation_forest.predict(scaled_features)

        # 4. 結果分析
        anomalies = []
        for i, (score, prediction) in enumerate(zip(anomaly_scores, anomaly_predictions)):
            if prediction == -1:  # 異常
                activity_data = current_activity[i]

                # 分類異常類型
                anomaly_type = await self.anomaly_classifier.classify_anomaly(
                    activity_data, score
                )

                anomaly_info = {
                    "activity_id": activity_data.get("id", f"activity_{i}"),
                    "anomaly_score": float(score),
                    "anomaly_type": anomaly_type,
                    "activity_data": activity_data,
                    "severity": self._calculate_anomaly_severity(score, anomaly_type),
                    "detected_at": datetime.now()
                }

                anomalies.append(anomaly_info)

        # 5. 異常聚合分析
        anomaly_clusters = await self._cluster_related_anomalies(anomalies)

        return {
            "anomalies": anomalies,
            "anomaly_clusters": anomaly_clusters,
            "normal_activities": len(current_activity) - len(anomalies),
            "anomaly_rate": len(anomalies) / len(current_activity),
            "severity_distribution": self._analyze_severity_distribution(anomalies)
        }

    async def _extract_security_features(self, activities: List[Dict]) -> np.ndarray:
        """提取安全相關特徵"""

        features = []

        for activity in activities:
            activity_features = []

            # 時間特徵
            timestamp = activity.get("timestamp", datetime.now())
            activity_features.extend([
                timestamp.hour,                    # 小時 (0-23)
                timestamp.weekday(),              # 星期 (0-6)
                (timestamp.hour >= 9 and timestamp.hour <= 17)  # 工作時間 (boolean -> int)
            ])

            # 用戶特徵
            user_info = activity.get("user", {})
            activity_features.extend([
                len(user_info.get("roles", [])),          # 角色數量
                len(user_info.get("departments", [])),    # 部門數量
                hash(user_info.get("location", "")) % 100 # 位置哈希
            ])

            # 訪問特徵
            access_info = activity.get("access", {})
            activity_features.extend([
                len(access_info.get("resources", [])),    # 訪問資源數量
                access_info.get("data_classification", 0), # 數據分類等級
                access_info.get("session_duration", 0)   # 會話持續時間
            ])

            # 查詢特徵
            query_info = activity.get("query", {})
            activity_features.extend([
                len(query_info.get("text", "")),         # 查詢長度
                query_info.get("complexity_score", 0),   # 查詢複雜度
                len(query_info.get("results", []))       # 結果數量
            ])

            features.append(activity_features)

        return np.array(features)

    def _calculate_anomaly_severity(self, score: float, anomaly_type: Dict) -> str:
        """計算異常嚴重程度"""

        base_severity = anomaly_type.get("base_severity", "medium")

        # 根據異常分數調整嚴重程度
        if score < -0.5:  # 高度異常
            if base_severity == "low":
                return "medium"
            elif base_severity == "medium":
                return "high"
            else:  # already high
                return "critical"
        elif score < -0.2:  # 中度異常
            return base_severity
        else:  # 輕度異常
            if base_severity == "high":
                return "medium"
            elif base_severity == "medium":
                return "low"
            else:
                return "low"

class SecurityEventResponseSystem:
    """安全事件回應系統"""

    def __init__(self):
        self.incident_manager = IncidentManager()
        self.response_playbooks = ResponsePlaybookManager()
        self.communication_system = SecurityCommunicationSystem()

    async def handle_security_event(self, event: Dict) -> Dict:
        """處理安全事件"""

        # 1. 事件分類和優先級分配
        event_classification = await self._classify_security_event(event)

        # 2. 選擇響應劇本
        response_playbook = await self.response_playbooks.select_playbook(
            event_classification
        )

        # 3. 自動化響應
        automated_response = await self._execute_automated_response(
            event, response_playbook
        )

        # 4. 人工介入判斷
        human_intervention = await self._assess_human_intervention_need(
            event, automated_response
        )

        # 5. 通知相關人員
        notification_result = await self.communication_system.notify_stakeholders(
            event, event_classification, human_intervention
        )

        return {
            "event_id": event.get("id", "unknown"),
            "classification": event_classification,
            "automated_response": automated_response,
            "human_intervention_required": human_intervention["required"],
            "notifications_sent": notification_result,
            "response_status": "handled"
        }
```

---

## 5. 本章總結

### 5.1 安全合規要點回顧

1. **零信任原則**: 永不信任、始終驗證、最小權限的系統化實現
2. **隱私保護**: 基於數學模型的 PII 檢測和智能匿名化
3. **合規自動化**: GDPR、SOC2、HIPAA 的技術實現框架
4. **安全監控**: 基於機器學習的異常檢測和自動化回應

### 5.2 實施最佳實踐

1. **安全優先設計**: 從系統設計初期就內建安全機制
2. **合規即代碼**: 將合規要求轉化為可自動執行的代碼
3. **持續監控**: 建立全方位的安全監控和告警體系
4. **事件準備**: 制定完整的安全事件回應預案

### 5.3 下章預告

第7章將探討 GraphRAG 和多智能體系統，這些先進技術為 RAG 系統帶來了新的安全挑戰和機遇，需要在安全框架中給予特殊考慮。

---

**課程評估**: 本章內容在期末考試中占15%權重，重點考查安全設計思維和合規實現能力。

**實作要求**: 學生需設計一個符合特定合規要求 (如 GDPR) 的 RAG 系統安全架構。