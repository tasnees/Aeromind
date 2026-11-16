import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TechnicalKnowledgeBase:
    """
    RAG Knowledge Base indexing turbofan technical documents and C-MAPSS literature.
    Extracts text, preserves page metadata, performs semantic chunking, and provides similarity retrieval.
    """

    def __init__(self, index_file: Path = Path("data/processed/knowledge_index.json")):
        self.index_file = Path(index_file)
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = None
        self.is_indexed = False

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

    def ingest_pdf(
        self,
        pdf_path: Path,
        document_id: str,
        document_title: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
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
        if not self.chunks:
            self.add_curated_knowledge()

        corpus = [c["text"] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

        # Save metadata index
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "w") as f:
            json.dump(self.chunks, f, indent=2)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.08,
    ) -> List[Dict[str, Any]]:
        """Performs semantic similarity search across technical document chunks."""
        if not self.is_indexed or self.tfidf_matrix is None:
            self.build_index()

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

