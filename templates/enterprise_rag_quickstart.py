#!/usr/bin/env python3
"""
Enterprise RAG Quick Start Template
Production-ready RAG system implementation

Usage:
    python enterprise_rag_quickstart.py --config config/production.yml
"""

import asyncio
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Core RAG components
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.vector_stores import QdrantVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama

# Document processing
from docling.document_converter import DocumentConverter
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Security & monitoring
from presidio_analyzer import AnalyzerEngine
import casbin
from ragas import evaluate
import opik

# Vector database
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

# Caching
import redis
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnterpriseRAGSystem:
    """Production-ready enterprise RAG system"""

    def __init__(self, config_path: str):
        """Initialize enterprise RAG system"""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.doc_processor = None
        self.vector_store = None
        self.llm = None
        self.embeddings = None
        self.index = None
        self.pii_detector = None
        self.access_control = None
        self.cache_client = None

        logger.info("Initializing Enterprise RAG System...")

    async def initialize(self):
        """Initialize all system components"""

        logger.info("🚀 Starting Enterprise RAG System initialization...")

        # Initialize document processor
        await self._initialize_document_processor()

        # Initialize vector database
        await self._initialize_vector_database()

        # Initialize embeddings
        await self._initialize_embeddings()

        # Initialize LLM
        await self._initialize_llm()

        # Initialize security components
        await self._initialize_security()

        # Initialize monitoring
        await self._initialize_monitoring()

        # Initialize caching
        await self._initialize_caching()

        logger.info("✅ Enterprise RAG System initialization completed!")

    async def _initialize_document_processor(self):
        """Initialize document processing pipeline"""

        logger.info("📄 Initializing document processor...")

        # Primary: Docling for advanced document processing
        self.doc_processor = DocumentConverter()

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['document_processing']['chunk_size'],
            chunk_overlap=self.config['document_processing']['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info("✅ Document processor initialized")

    async def _initialize_vector_database(self):
        """Initialize Qdrant vector database"""

        logger.info("🔍 Initializing vector database...")

        # Connect to Qdrant
        qdrant_config = self.config['vector_database']['qdrant']
        self.qdrant_client = QdrantClient(
            host=qdrant_config['host'],
            port=qdrant_config['port'],
            prefer_grpc=True
        )

        # Create collection if it doesn't exist
        collection_name = qdrant_config['collection_name']

        try:
            collections = await self.qdrant_client.get_collections()
            collection_exists = any(
                collection.name == collection_name
                for collection in collections.collections
            )

            if not collection_exists:
                await self._create_production_collection(collection_name)

        except Exception as e:
            logger.warning(f"Collection check failed: {e}")
            await self._create_production_collection(collection_name)

        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name
        )

        logger.info("✅ Vector database initialized")

    async def _create_production_collection(self, collection_name: str):
        """Create production-optimized Qdrant collection"""

        logger.info(f"Creating collection: {collection_name}")

        await self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024,  # BGE-large embedding size
                distance=Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=64,              # Higher connectivity for accuracy
                    ef_construct=256,  # Build quality
                    full_scan_threshold=10000
                )
            ),
            # Production optimization
            optimizers_config=models.OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=8,
                max_segment_size=100000
            ),
            # Sharding for scale
            shard_number=3,
            replication_factor=1
        )

    async def _initialize_embeddings(self):
        """Initialize embedding models"""

        logger.info("🧮 Initializing embedding models...")

        # Primary embedding model
        embedding_config = self.config['embeddings']
        self.embeddings = HuggingFaceEmbedding(
            model_name=embedding_config['primary_model'],
            device='cuda' if embedding_config.get('use_gpu', True) else 'cpu',
            normalize=True
        )

        logger.info("✅ Embeddings initialized")

    async def _initialize_llm(self):
        """Initialize LLM for generation"""

        logger.info("🤖 Initializing LLM...")

        llm_config = self.config['llm']

        if llm_config['provider'] == 'ollama':
            self.llm = Ollama(
                model=llm_config['model'],
                base_url=llm_config['base_url'],
                temperature=llm_config.get('temperature', 0.1)
            )

        logger.info("✅ LLM initialized")

    async def _initialize_security(self):
        """Initialize security components"""

        logger.info("🔒 Initializing security components...")

        # PII detection
        self.pii_detector = AnalyzerEngine()

        # Access control
        security_config = self.config['security']
        self.access_control = casbin.Enforcer(
            security_config['rbac_model'],
            security_config['rbac_policy']
        )

        logger.info("✅ Security components initialized")

    async def _initialize_monitoring(self):
        """Initialize monitoring and evaluation"""

        logger.info("📊 Initializing monitoring...")

        # Initialize Opik if configured
        monitoring_config = self.config.get('monitoring', {})

        if monitoring_config.get('opik_enabled', False):
            import opik
            self.opik_client = opik.Opik()

        logger.info("✅ Monitoring initialized")

    async def _initialize_caching(self):
        """Initialize caching layer"""

        logger.info("⚡ Initializing caching...")

        cache_config = self.config['caching']
        self.cache_client = redis.Redis(
            host=cache_config['host'],
            port=cache_config['port'],
            db=cache_config['db'],
            decode_responses=True
        )

        # Test cache connection
        try:
            await self.cache_client.ping()
            logger.info("✅ Cache initialized and connected")
        except Exception as e:
            logger.warning(f"Cache connection failed: {e}")

    async def process_documents(self, documents_path: str) -> Dict:
        """Process documents and build index"""

        logger.info(f"📚 Processing documents from: {documents_path}")

        # Load documents
        reader = SimpleDirectoryReader(documents_path)
        documents = reader.load_data()

        processed_docs = []

        for doc in documents:
            # Advanced document processing with Docling
            docling_result = self.doc_processor.convert(doc.text)

            # PII detection and protection
            pii_results = self.pii_detector.analyze(
                text=docling_result.document.export_to_markdown(),
                language="en"
            )

            # Check if document contains sensitive information
            if pii_results:
                logger.warning(f"PII detected in document: {doc.doc_id}")
                # In production, implement anonymization

            # Chunk document
            chunks = self.text_splitter.split_text(
                docling_result.document.export_to_markdown()
            )

            for chunk in chunks:
                processed_docs.append({
                    "text": chunk,
                    "metadata": {
                        "source_document": doc.doc_id,
                        "processing_timestamp": datetime.now().isoformat(),
                        "pii_detected": len(pii_results) > 0
                    }
                })

        # Create service context
        service_context = ServiceContext.from_defaults(
            embed_model=self.embeddings,
            llm=self.llm,
            chunk_size=self.config['document_processing']['chunk_size']
        )

        # Build index
        logger.info("🏗️ Building vector index...")
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            service_context=service_context
        )

        # Add documents to index
        for doc_data in processed_docs:
            self.index.insert(doc_data["text"], metadata=doc_data["metadata"])

        logger.info(f"✅ Processed {len(processed_docs)} document chunks")

        return {
            "documents_processed": len(documents),
            "chunks_created": len(processed_docs),
            "index_built": True
        }

    async def query(self, query_text: str, user_context: Dict = None) -> Dict:
        """Execute RAG query with enterprise features"""

        start_time = datetime.now()

        # Check cache first
        cache_key = self._generate_cache_key(query_text, user_context)
        cached_result = await self._get_cached_result(cache_key)

        if cached_result:
            logger.info(f"📋 Cache hit for query: {query_text[:50]}...")
            return cached_result

        # Security checks
        security_check = await self._perform_security_checks(query_text, user_context)
        if not security_check['authorized']:
            return {
                "error": "Access denied",
                "reason": security_check['reason'],
                "timestamp": start_time.isoformat()
            }

        # Execute query
        logger.info(f"🔍 Processing query: {query_text[:50]}...")

        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.config['retrieval']['top_k'],
            response_mode="tree_summarize"  # Better for long contexts
        )

        # Execute query
        query_result = await query_engine.aquery(query_text)

        # Extract sources
        source_nodes = query_result.source_nodes
        sources = [
            {
                "content": node.text,
                "metadata": node.metadata,
                "score": node.score
            }
            for node in source_nodes
        ]

        # Generate response
        response = {
            "answer": str(query_result),
            "sources": sources,
            "query": query_text,
            "timestamp": start_time.isoformat(),
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "num_sources": len(sources)
        }

        # Cache result
        await self._cache_result(cache_key, response)

        # Log for monitoring
        await self._log_query_execution(query_text, response, user_context)

        logger.info(f"✅ Query completed in {response['processing_time_ms']:.1f}ms")

        return response

    async def _perform_security_checks(self, query: str, user_context: Dict = None) -> Dict:
        """Perform security and access control checks"""

        if not user_context:
            return {"authorized": False, "reason": "No user context provided"}

        # Check user authorization
        user_id = user_context.get('user_id', 'anonymous')
        resource = "rag_system"
        action = "query"

        authorized = self.access_control.enforce(user_id, resource, action)

        # Additional security checks
        security_checks = {
            "user_authorized": authorized,
            "query_safe": await self._check_query_safety(query),
            "rate_limit_ok": await self._check_rate_limit(user_id)
        }

        all_checks_passed = all(security_checks.values())

        return {
            "authorized": all_checks_passed,
            "reason": "Access granted" if all_checks_passed else "Security check failed",
            "checks": security_checks
        }

    async def _check_query_safety(self, query: str) -> bool:
        """Check query for safety issues"""

        # Simple prompt injection detection
        suspicious_patterns = [
            "ignore previous instructions",
            "act as",
            "pretend you are",
            "system:",
            "<script>",
            "exec(",
            "eval("
        ]

        query_lower = query.lower()
        return not any(pattern in query_lower for pattern in suspicious_patterns)

    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check user rate limit"""

        if not self.cache_client:
            return True  # No rate limiting if cache unavailable

        try:
            # Check request count in last minute
            rate_limit_key = f"rate_limit:{user_id}"
            current_count = await self.cache_client.get(rate_limit_key) or 0

            max_requests_per_minute = self.config['security'].get('max_requests_per_minute', 60)

            if int(current_count) >= max_requests_per_minute:
                return False

            # Increment counter
            await self.cache_client.incr(rate_limit_key)
            await self.cache_client.expire(rate_limit_key, 60)  # 1 minute TTL

            return True

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True  # Allow on error

    def _generate_cache_key(self, query: str, user_context: Dict = None) -> str:
        """Generate cache key for query"""

        key_components = [query]

        if user_context:
            # Include user department for access-aware caching
            if 'department' in user_context:
                key_components.append(f"dept:{user_context['department']}")

        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached query result"""

        if not self.cache_client:
            return None

        try:
            cached_data = await self.cache_client.get(f"query_result:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    async def _cache_result(self, cache_key: str, result: Dict):
        """Cache query result"""

        if not self.cache_client:
            return

        try:
            ttl = self.config['caching'].get('query_ttl', 3600)  # 1 hour default
            await self.cache_client.setex(
                f"query_result:{cache_key}",
                ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    async def _log_query_execution(self, query: str, response: Dict, user_context: Dict = None):
        """Log query execution for monitoring"""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "processing_time_ms": response["processing_time_ms"],
            "num_sources": response["num_sources"],
            "user_id": user_context.get('user_id', 'anonymous') if user_context else 'anonymous',
            "success": True
        }

        # Log to structured logger
        logger.info("Query executed", extra=log_entry)

        # Send to monitoring system if configured
        if hasattr(self, 'opik_client'):
            try:
                await self._send_to_opik(log_entry)
            except Exception as e:
                logger.warning(f"Opik logging failed: {e}")

    async def _send_to_opik(self, log_entry: Dict):
        """Send metrics to Opik monitoring"""

        # Implementation would send to actual Opik instance
        pass

    async def evaluate_system(self, test_queries: List[Dict]) -> Dict:
        """Evaluate system performance using RAGAS"""

        logger.info("🧪 Starting system evaluation...")

        evaluation_data = []

        for test_case in test_queries:
            query = test_case['query']
            expected_answer = test_case.get('expected_answer', '')

            # Execute query
            result = await self.query(query)

            if 'error' not in result:
                evaluation_data.append({
                    "question": query,
                    "answer": result["answer"],
                    "contexts": [source["content"] for source in result["sources"]],
                    "ground_truth": expected_answer
                })

        if not evaluation_data:
            logger.warning("No valid evaluation data available")
            return {"error": "No valid queries for evaluation"}

        # Run RAGAS evaluation
        from datasets import Dataset

        dataset = Dataset.from_list(evaluation_data)

        # Configure RAGAS metrics
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )

            logger.info("✅ System evaluation completed")

            return {
                "evaluation_results": dict(evaluation_result),
                "test_cases_evaluated": len(evaluation_data),
                "overall_score": sum(evaluation_result.values()) / len(evaluation_result)
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": f"Evaluation failed: {str(e)}"}

    async def health_check(self) -> Dict:
        """Comprehensive system health check"""

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }

        # Check vector database
        try:
            collections = await self.qdrant_client.get_collections()
            health_status["components"]["qdrant"] = {
                "status": "healthy",
                "collections_count": len(collections.collections)
            }
        except Exception as e:
            health_status["components"]["qdrant"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"

        # Check LLM
        try:
            test_response = await self.llm.acomplete("Test")
            health_status["components"]["llm"] = {
                "status": "healthy",
                "test_response_length": len(str(test_response))
            }
        except Exception as e:
            health_status["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"

        # Check cache
        if self.cache_client:
            try:
                await self.cache_client.ping()
                health_status["components"]["cache"] = {"status": "healthy"}
            except Exception as e:
                health_status["components"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        return health_status


async def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Enterprise RAG Quick Start")
    parser.add_argument(
        "--config",
        default="configs/quickstart_config.yml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--documents",
        default="./sample_documents",
        help="Documents directory to process"
    )
    parser.add_argument(
        "--test-queries",
        default="./test_queries.json",
        help="Test queries file for evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["initialize", "process", "query", "evaluate", "health"],
        default="initialize",
        help="Operation mode"
    )

    args = parser.parse_args()

    # Initialize RAG system
    rag_system = EnterpriseRAGSystem(args.config)

    if args.mode == "initialize":
        await rag_system.initialize()
        logger.info("🎉 Enterprise RAG system initialized successfully!")

        # Run health check
        health_status = await rag_system.health_check()
        logger.info(f"System health: {health_status['overall_status']}")

    elif args.mode == "process":
        await rag_system.initialize()

        if Path(args.documents).exists():
            result = await rag_system.process_documents(args.documents)
            logger.info(f"📚 Document processing completed: {result}")
        else:
            logger.error(f"Documents directory not found: {args.documents}")

    elif args.mode == "query":
        await rag_system.initialize()

        # Interactive query mode
        print("\n🤖 Enterprise RAG System - Interactive Mode")
        print("Enter your queries (type 'exit' to quit):\n")

        while True:
            query = input("Query: ").strip()

            if query.lower() == 'exit':
                break

            if query:
                result = await rag_system.query(
                    query,
                    user_context={"user_id": "demo_user", "department": "general"}
                )

                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n📝 Answer: {result['answer']}")
                    print(f"🔍 Sources: {result['num_sources']}")
                    print(f"⚡ Time: {result['processing_time_ms']:.1f}ms\n")

    elif args.mode == "evaluate":
        await rag_system.initialize()

        if Path(args.test_queries).exists():
            import json
            with open(args.test_queries, 'r') as f:
                test_queries = json.load(f)

            result = await rag_system.evaluate_system(test_queries)
            logger.info(f"📊 Evaluation completed: {result}")
        else:
            logger.error(f"Test queries file not found: {args.test_queries}")

    elif args.mode == "health":
        await rag_system.initialize()
        health_status = await rag_system.health_check()

        print(f"\n🏥 System Health Check")
        print(f"Overall Status: {health_status['overall_status']}")
        print(f"Timestamp: {health_status['timestamp']}")

        for component, status in health_status['components'].items():
            status_emoji = "✅" if status['status'] == 'healthy' else "❌"
            print(f"{status_emoji} {component}: {status['status']}")

            if status['status'] == 'unhealthy':
                print(f"   Error: {status.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())