"""RAG (Retrieval-Augmented Generation) tool for MAGI-02.

This module provides a local RAG implementation using ChromaDB for vector storage
and sentence-transformers for embeddings.
"""

import argparse
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from strands import tool

from magi.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)

# Global instances for reuse
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or create the embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB client singleton."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_or_create_collection() -> chromadb.Collection:
    """Get or create the document collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "MAGI-02 document collection"},
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def read_file_content(file_path: Path) -> str:
    """Read content from a file based on its extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("pypdf is required to read PDF files. Install with: pip install pypdf")
    elif suffix in [".txt", ".md", ".rst", ".py", ".js", ".json", ".yaml", ".yml"]:
        return file_path.read_text(encoding="utf-8")
    else:
        # Try to read as text
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"Cannot read file {file_path}: unsupported format")


def ingest_documents(documents_dir: Path = DOCUMENTS_DIR) -> dict:
    """Ingest documents from the documents directory into the vector store.

    Args:
        documents_dir: Path to the directory containing documents

    Returns:
        Dictionary with ingestion statistics
    """
    if not documents_dir.exists():
        return {"status": "error", "message": f"Documents directory not found: {documents_dir}"}

    collection = get_or_create_collection()
    model = get_embedding_model()

    stats = {"files_processed": 0, "chunks_added": 0, "errors": []}

    # Get all files in the documents directory
    files = list(documents_dir.glob("**/*"))
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    for file_path in files:
        try:
            content = read_file_content(file_path)
            chunks = chunk_text(content)

            # Generate embeddings
            embeddings = model.encode(chunks).tolist()

            # Create unique IDs for each chunk using relative path to avoid collisions
            relative_path = file_path.relative_to(documents_dir)
            safe_path = str(relative_path).replace("/", "_").replace("\\", "_")
            ids = [f"{safe_path}_{i}" for i in range(len(chunks))]

            # Create metadata
            metadatas = [
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            stats["files_processed"] += 1
            stats["chunks_added"] += len(chunks)

        except Exception as e:
            stats["errors"].append({"file": str(file_path), "error": str(e)})

    return stats


@tool
def rag_search(query: str, top_k: int = TOP_K_RESULTS) -> str:
    """Search through local documents using semantic search.

    This tool searches through indexed documents using vector similarity
    to find relevant information for answering questions.

    Args:
        query: The search query to find relevant documents
        top_k: Number of top results to return (default: 3)

    Returns:
        A formatted string containing relevant document excerpts with sources
    """
    collection = get_or_create_collection()
    model = get_embedding_model()

    # Check if collection has documents
    if collection.count() == 0:
        return "No documents have been indexed. Please add documents to the data/documents folder and run ingestion."

    # Generate query embedding
    query_embedding = model.encode([query]).tolist()

    # Search the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant documents found for your query."

    # Format results
    output_parts = ["## Relevant Document Excerpts\n"]

    for i, (doc, metadata, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        source = metadata.get("filename", "Unknown source")
        similarity = 1 - distance  # Convert distance to similarity score
        output_parts.append(f"### Source {i + 1}: {source} (relevance: {similarity:.2f})\n")
        output_parts.append(f"{doc}\n")

    return "\n".join(output_parts)


def clear_collection() -> dict:
    """Clear all documents from the collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        return {"status": "success", "message": "Collection cleared successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAGI RAG Tool")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents from the documents directory",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all documents from the collection",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search for documents matching the query",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DOCUMENTS_DIR),
        help="Directory containing documents to ingest",
    )

    args = parser.parse_args()

    if args.ingest:
        print(f"Ingesting documents from {args.dir}...")
        result = ingest_documents(Path(args.dir))
        print(f"Result: {result}")
    elif args.clear:
        print("Clearing document collection...")
        result = clear_collection()
        print(f"Result: {result}")
    elif args.search:
        print(f"Searching for: {args.search}")
        result = rag_search(args.search)
        print(result)
    else:
        parser.print_help()
