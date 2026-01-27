#!/usr/bin/env python3
# Build FAISS index for the DGM Study Assistant

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dgm_study_assistant.rag.loader import (
    load_slides,
    load_transcripts,
    split_docs,
    get_embeddings,
    build_vectorstore
)

def main():
    # Load all course materials
    print("Loading documents...")
    transcript_docs = load_transcripts()
    slides_docs = load_slides()
    
    all_docs = transcript_docs + slides_docs
    print(f"Loaded {len(all_docs)} documents total")
    
    # Chunk them for better retrieval
    print("\nSplitting into chunks...")
    chunks = split_docs(all_docs)
    print(f"Created {len(chunks)} chunks")
    
    # Build the index
    print("\nBuilding FAISS index...")
    embedding_model = get_embeddings()
    save_path = Path(__file__).parent / "faiss_index"
    
    vectorstore = build_vectorstore(
        chunks,
        embedding_model,
        save_path=str(save_path),
        batch_size=32
    )
    
    print(f"Done! Index saved to {save_path}")
    
    # Quick test
    print("\nTesting with 'What is a VAE?'...")
    results = vectorstore.similarity_search("What is a VAE?", k=2)
    for doc in results:
        source = doc.metadata.get('source', 'unknown')
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  - {source}: {preview}...")

if __name__ == "__main__":
    main()
