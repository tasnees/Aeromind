"""
Knowledge Base Builder for AeroMind Ops.

Ingests all available technical documents into the Qdrant vector store:
  1. Damage Propagation Modeling PDF (C-MAPSS NASA technical paper)
  2. C-MAPSS readme / operational notes
  3. Curated engineering specifications and maintenance SOPs

Usage:
    python -m aeromind.rag.build_kb
    python -m aeromind.rag.build_kb --pdf data/raw/custom_manual.pdf --title "Custom Manual"
    python -m aeromind.rag.build_kb --verify   # rebuild + run search smoke test
"""

import argparse
import json
import time
from pathlib import Path

from aeromind.rag.knowledge_base import TechnicalKnowledgeBase


def ingest_text_file(kb: TechnicalKnowledgeBase, path: Path, doc_id: str, title: str, chunk_size: int = 400, chunk_overlap: int = 50):
    """Ingests a plain-text file by sliding-window chunking (PDF fallback for .txt files)."""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        print(f"  [skip] {path.name} is empty.")
        return 0

    words = text.split()
    step = max(1, chunk_size - chunk_overlap)
    added = 0
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        if len(chunk_words) < 20:          # skip tiny trailing fragments
            continue
        chunk_text = " ".join(chunk_words)
        kb.chunks.append({
            "document_id": doc_id,
            "document_title": title,
            "page": i // step + 1,
            "chunk_id": f"{doc_id}_c{i // step}",
            "source_type": "technical_document",
            "text": chunk_text,
        })
        added += 1
    return added


def build_knowledge_base(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    verify: bool = False,
) -> TechnicalKnowledgeBase:
    """
    Full knowledge base ingestion pipeline:
      Step 1 — Reset existing index
      Step 2 — Ingest PDF (Damage Propagation Modeling) via PyPDF2 if available
      Step 3 — Ingest readme.txt operational notes
      Step 4 — Add curated engineering + maintenance SOP entries
      Step 5 — Build Qdrant + TF-IDF index
      Step 6 — Persist JSON cache
    """
    print("=" * 60)
    print("AeroMind Ops — Knowledge Base Builder")
    print("=" * 60)
    t0 = time.time()

    index_file = processed_dir / "knowledge_index.json"
    kb = TechnicalKnowledgeBase(
        index_file=index_file,
        collection_name="aeromind_technical_docs",
        vector_dim=384,
    )

    # Reset — rebuild from scratch
    kb.chunks = []

    total_chunks = 0

    # ------------------------------------------------------------------ #
    # Step 2: Ingest C-MAPSS NASA Technical Paper (PDF)
    # ------------------------------------------------------------------ #
    pdf_path = raw_dir / "Damage Propagation Modeling.pdf"
    if pdf_path.exists():
        try:
            from aeromind.rag.knowledge_base import PYPDF2_AVAILABLE
            if PYPDF2_AVAILABLE:
                print(f"\n[PDF] Ingesting: {pdf_path.name}")
                kb.ingest_pdf(
                    pdf_path=pdf_path,
                    document_id="damage_propagation_model",
                    document_title="Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation (NASA/CMAPSS)",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                added = len(kb.chunks)
                total_chunks += added
                print(f"  ✓ {added} chunks extracted from PDF.")
            else:
                print(f"\n[PDF] PyPDF2 not installed — falling back to text extraction not available.")
                print("  Install PyPDF2 with: pip install PyPDF2")
        except Exception as e:
            print(f"  [Warning] PDF ingestion error: {e}")
    else:
        print(f"\n[PDF] Not found: {pdf_path}  (skipping)")

    # ------------------------------------------------------------------ #
    # Step 3: Ingest readme.txt — C-MAPSS operational metadata
    # ------------------------------------------------------------------ #
    readme_path = raw_dir / "readme.txt"
    if readme_path.exists():
        print(f"\n[TXT] Ingesting: {readme_path.name}")
        n = ingest_text_file(
            kb, readme_path,
            doc_id="cmapss_readme",
            title="C-MAPSS Dataset Description and Sensor Reference (NASA)",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        total_chunks += n
        print(f"  ✓ {n} chunks extracted from readme.")

    # ------------------------------------------------------------------ #
    # Step 4: Curated engineering + maintenance SOP knowledge
    # ------------------------------------------------------------------ #
    print("\n[CURATED] Adding engineering specifications and maintenance SOPs...")
    before = len(kb.chunks)
    kb.add_curated_knowledge()
    n_curated = len(kb.chunks) - before
    total_chunks += n_curated
    print(f"  ✓ {n_curated} curated knowledge entries added.")

    # ------------------------------------------------------------------ #
    # Step 5: Build Qdrant + TF-IDF index
    # ------------------------------------------------------------------ #
    print(f"\n[INDEX] Building vector index for {len(kb.chunks)} total chunks...")
    kb.build_index()
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Knowledge Base Built Successfully")
    print(f"  Total chunks indexed : {len(kb.chunks)}")
    print(f"  Qdrant collection    : {kb.collection_name}")
    print(f"  Index cache file     : {index_file}")
    print(f"  Time elapsed         : {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------ #
    # Step 6: Optional smoke test
    # ------------------------------------------------------------------ #
    if verify:
        print("\n[VERIFY] Running search smoke tests...")
        test_queries = [
            "HPC blade degradation high pressure compressor inspection",
            "remaining useful life below 30 cycles critical maintenance action",
            "FD001 sea level single fault operating condition sensor",
            "turbofan engine borescope compressor blade crack",
        ]
        all_passed = True
        for q in test_queries:
            results = kb.search(q, top_k=2)
            status = "✓ PASS" if results else "✗ FAIL"
            if not results:
                all_passed = False
            print(f"  {status} | '{q[:55]}...' → {len(results)} result(s)")
            if results:
                print(f"           Top: '{results[0]['document_title']}' (score={results[0].get('similarity_score', '?')})")

        print(f"\nSmoke test: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")

    return kb


def save_kb_manifest(kb: TechnicalKnowledgeBase, out_path: Path = Path("data/processed/kb_manifest.json")):
    """Saves a summary manifest of the indexed knowledge base."""
    from collections import Counter
    source_counts = Counter(c["source_type"] for c in kb.chunks)
    doc_counts = Counter(c["document_title"] for c in kb.chunks)
    manifest = {
        "total_chunks": len(kb.chunks),
        "collection_name": kb.collection_name,
        "vector_dim": kb.vector_dim,
        "chunks_by_source_type": dict(source_counts),
        "chunks_by_document": dict(doc_counts),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Manifest] Saved to {out_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroMind Knowledge Base Builder")
    parser.add_argument("--raw-dir",    type=str, default="data/raw",       help="Directory containing raw PDFs and text files")
    parser.add_argument("--out-dir",    type=str, default="data/processed", help="Directory for index output files")
    parser.add_argument("--chunk-size", type=int, default=400,              help="Words per chunk (default 400)")
    parser.add_argument("--overlap",    type=int, default=50,               help="Overlap words between chunks (default 50)")
    parser.add_argument("--verify",     action="store_true",                help="Run search smoke test after building")
    parser.add_argument("--pdf",        type=str, default=None,             help="Path to additional PDF to ingest")
    parser.add_argument("--title",      type=str, default="Additional Technical Manual", help="Title for --pdf document")
    args = parser.parse_args()

    kb = build_knowledge_base(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.out_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        verify=args.verify,
    )

    # Ingest any extra PDF passed on the command line
    if args.pdf:
        extra_pdf = Path(args.pdf)
        if extra_pdf.exists():
            print(f"\n[EXTRA PDF] Ingesting: {extra_pdf.name}")
            kb.ingest_pdf(
                pdf_path=extra_pdf,
                document_id=extra_pdf.stem.lower().replace(" ", "_"),
                document_title=args.title,
            )
            kb.build_index()     # re-index with new document
            print(f"  ✓ Extra PDF indexed. Total chunks: {len(kb.chunks)}")
        else:
            print(f"  [Warning] Extra PDF not found: {extra_pdf}")

    save_kb_manifest(kb)
