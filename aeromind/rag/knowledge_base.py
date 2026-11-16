"""
Technical Knowledge Base & Vector Retrieval Engine for AeroMind Ops.

Primary Target Vector Database: Qdrant (FR-LLM-02, FR-BE-02)
- Embedded in-process mode (data/processed/qdrant_storage) for local development and CI/CD.
- Client-Server mode (via QDRANT_URL e.g. http://qdrant:6333) for containerized deployments.
- High-performance HNSW cosine distance indexing with rich payload metadata filtering.
- Automated fallback to local sparse vector store if qdrant-client is not yet installed.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import PyPDF2  # type: ignore
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qmodels  # type: ignore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DenseVectorEncoder:
    """
    Semantic dense vector encoder (384-dim standard).
    Uses sentence-transformers if available, or deterministic normalized dense projection.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._st_model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._st_model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._st_model is not None:
            return self._st_model.encode(texts, convert_to_numpy=True)

        # High-entropy deterministic dense vector projection (384 dimensions)
        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self.dim, dtype=np.float32)
            if not words:
                vectors.append(vec)
                continue

            for word in words:
                # Deterministic hashing into 384-dimensional space
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
                vec[idx] += sign

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)


class TechnicalKnowledgeBase:
    """
    Enterprise RAG Knowledge Base powered by Qdrant Vector Database.
    Extracts text, chunks engineering manuals, computes semantic embeddings,
    and conducts fast similarity retrieval with payload metadata filtering.
    """

    def __init__(
        self,
        index_file: Path = Path("data/processed/knowledge_index.json"),
        collection_name: str = "aeromind_technical_docs",
        vector_dim: int = 384,
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[Path] = Path("data/processed/qdrant_storage"),
    ):
        self.index_file = Path(index_file)
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.chunks: List[Dict[str, Any]] = []

        # Local TF-IDF Fallback
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = None
        self.is_indexed = False

        # Dense Encoder
        self.dense_encoder = DenseVectorEncoder(dim=self.vector_dim)

        # Initialize Qdrant Client
        self.use_qdrant = QDRANT_AVAILABLE
        self.qdrant_client = None

        if self.use_qdrant:
            url = qdrant_url or os.getenv("QDRANT_URL")
            try:
                if url:
                    self.qdrant_client = QdrantClient(url=url, timeout=5.0)
                    print(f"[KnowledgeBase] Connected to Qdrant server at {url}")
                else:
                    path = Path(qdrant_path) if qdrant_path else Path("data/processed/qdrant_storage")
                    path.mkdir(parents=True, exist_ok=True)
                    self.qdrant_client = QdrantClient(path=str(path))
                    print(f"[KnowledgeBase] Initialized embedded Qdrant database at {path}")

                self._ensure_qdrant_collection()
            except Exception as e:
                print(f"[KnowledgeBase] Qdrant initialization note: {e}. Falling back to in-memory store.")
                self.use_qdrant = False
        else:
            print("[KnowledgeBase] Qdrant client not installed. Using local sparse vector store. (Run 'pip install qdrant-client' to enable Qdrant).")

        # Load pre-built index if present on disk
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                if self.chunks:
                    corpus = [c["text"] for c in self.chunks]
                    self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
                    self.is_indexed = True
            except Exception as e:
                print(f"[KnowledgeBase] Warning loading existing index: {e}")

    def _ensure_qdrant_collection(self):
        """Ensures the Qdrant collection exists with the required vector schema."""
        if not self.use_qdrant or not self.qdrant_client:
            return

        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_dim,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                print(f"[KnowledgeBase] Created Qdrant collection '{self.collection_name}' (size={self.vector_dim}, distance=COSINE).")
        except Exception as e:
            print(f"[KnowledgeBase] Warning ensuring Qdrant collection: {e}")

    def ingest_pdf(
        self,
        pdf_path: Path,
        document_id: str,
        document_title: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Ingests a technical PDF document and performs sliding-window chunking."""
        if not PYPDF2_AVAILABLE:
            print(f"[KnowledgeBase] Warning: PyPDF2 is not installed. Skipping PDF ingestion for '{document_title}'.")
            return

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PyPDF2.PdfReader(str(pdf_path))
        num_pages = len(reader.pages)

        for page_num in range(num_pages):
            page_text = reader.pages[page_num].extract_text()
            if not page_text or len(page_text.strip()) < 50:
                continue

            words = page_text.split()
            step = chunk_size - chunk_overlap
            for i in range(0, len(words), step):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words)

                chunk_record = {
                    "document_id": document_id,
                    "document_title": document_title,
                    "page": page_num + 1,
                    "chunk_id": f"{document_id}_p{page_num+1}_c{i//step}",
                    "source_type": "technical_document",
                    "text": chunk_text,
                }
                self.chunks.append(chunk_record)

    def add_curated_knowledge(self):
        """Adds curated C-MAPSS technical specifications and operational rules."""
        curated_entries = [
            {
                "document_id": "cmapss_spec_001",
                "document_title": "C-MAPSS Simulation Reference Manual",
                "page": 1,
                "chunk_id": "cmapss_spec_c1",
                "source_type": "engineering_specification",
                "text": "The C-MAPSS turbofan model simulates a 90,000 lb thrust class engine. HPC (High Pressure Compressor) fault mode is characterized by degradation in flow capacity and adiabatic efficiency. Key indicators include rising HPC outlet temperature (T50 / sensor_3), static pressure drop (Ps30 / sensor_11), and elevated core speed (Nc / sensor_9).",
            },
            {
                "document_id": "cmapss_spec_002",
                "document_title": "C-MAPSS Fault Modes & Operational Limits",
                "page": 2,
                "chunk_id": "cmapss_spec_c2",
                "source_type": "engineering_specification",
                "text": "Sub-dataset FD001 simulates sea-level standard day operating conditions with single fault mode HPC degradation. Sub-dataset FD004 introduces six flight conditions spanning altitudes from 0 to 42,000 ft, Mach numbers from 0.0 to 0.84, and dual fault modes combining HPC and Fan degradation. HPC coolant bleed (sensor_20) and LPT coolant bleed (sensor_21) reflect turbine thermal load compensation.",
            },
            {
                "document_id": "maintenance_proc_042",
                "document_title": "Turbofan Predictive Maintenance Standard Operating Procedures",
                "page": 14,
                "chunk_id": "maint_proc_c1",
                "source_type": "maintenance_sop",
                "text": "When Remaining Useful Life (RUL) drops below 30 cycles (Critical Degradation Stage), immediate borescope inspection of the High Pressure Compressor (HPC) stage 3-5 blades is mandatory within 5 cycles. For RUL between 30 and 80 cycles (Degraded Stage), schedule compressor blade tip clearance monitoring and trend analysis.",
            },
        ]
        self.chunks.extend(curated_entries)

    def build_index(self):
        """
        Builds the vector index in Qdrant (and syncs local cache).
        """
        if not self.chunks:
            self.add_curated_knowledge()

        corpus = [c["text"] for c in self.chunks]

        # 1. Index in Qdrant if active
        if self.use_qdrant and self.qdrant_client:
            try:
                self._ensure_qdrant_collection()
                dense_vectors = self.dense_encoder.encode(corpus)
                points = []
                for idx, (chunk, vec) in enumerate(zip(self.chunks, dense_vectors)):
                    points.append(
                        qmodels.PointStruct(
                            id=idx,
                            vector=vec.tolist(),
                            payload=chunk,
                        )
                    )

                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                print(f"[KnowledgeBase] Successfully indexed {len(points)} chunks into Qdrant collection '{self.collection_name}'.")
            except Exception as e:
                print(f"[KnowledgeBase] Error upserting to Qdrant: {e}")

        # 2. Local TF-IDF Fallback Matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

        # 3. Persist JSON index cache
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.08,
        filter_source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search.
        Queries Qdrant when available, falling back to local cosine similarity.
        """
        if not self.is_indexed:
            self.build_index()

        # 1. Search in Qdrant
        if self.use_qdrant and self.qdrant_client:
            try:
                query_vec = self.dense_encoder.encode([query])[0].tolist()
                
                query_filter = None
                if filter_source_type:
                    query_filter = qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="source_type",
                                match=qmodels.MatchValue(value=filter_source_type),
                            )
                        ]
                    )

                hits = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=top_k,
                    score_threshold=min_similarity,
                    query_filter=query_filter,
                )

                if hits:
                    results = []
                    for hit in hits:
                        record = dict(hit.payload)
                        record["similarity_score"] = round(float(hit.score), 4)
                        results.append(record)
                    return results
            except Exception as e:
                print(f"[KnowledgeBase] Qdrant search fallback: {e}")

        # 2. Local Sparse TF-IDF Cosine Similarity Fallback
        if self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1]
        results = []

        for idx in top_indices[:top_k]:
            sim = float(similarities[idx])
            if sim >= min_similarity:
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = round(sim, 4)
                results.append(chunk)

        return results
