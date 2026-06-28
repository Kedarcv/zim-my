"""
Zim-my RAG Pipeline
===================
ChromaDB + sentence-transformers retrieval over Shona/English agriculture corpus.

Model: Zim-my (fine-tuned Qwen2.5-3B-Instruct)
Developer: Michael Mlungisi Nkomo — AI Engineer from Zimbabwe

Usage:
    from rag import ZimMyRAG
    rag = ZimMyRAG()
    rag.index_corpus("data/rag/zimbabwe_agriculture.jsonl")
    context = rag.query("What crops grow in Mashonaland?")
"""

import json
import os
from pathlib import Path
from typing import List, Optional

# Embedding model — small enough for 7GB RAM budget
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80MB, 384-dim vectors
CHROMA_PERSIST_DIR = "data/rag/chroma_db"
COLLECTION_NAME = "zim_my_agriculture"


class ZimMyRAG:
    """Retrieval-Augmented Generation pipeline for Zim-my."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.embedding_model_name = embedding_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Lazy imports — only load when needed to save RAM
        self._chroma_client = None
        self._collection = None
        self._embedder = None

    def _init_embedder(self):
        """Initialize sentence-transformers embedder."""
        if self._embedder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("Installing sentence-transformers...")
            os.system("pip install sentence-transformers")
            from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {self.embedding_model_name}...")
        self._embedder = SentenceTransformer(self.embedding_model_name)
        print("✓ Embedding model loaded")

    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        if self._chroma_client is not None:
            return

        try:
            import chromadb
        except ImportError:
            print("Installing chromadb...")
            os.system("pip install chromadb")
            import chromadb

        self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✓ ChromaDB collection '{self.collection_name}' ready "
              f"({self._collection.count()} documents)")

    def index_corpus(
        self,
        data_path: str,
        text_field: str = "text",
        id_field: str = "id",
        metadata_fields: Optional[List[str]] = None,
        batch_size: int = 500,
    ):
        """
        Index a JSONL corpus into ChromaDB.

        Args:
            data_path: Path to JSONL file with one JSON object per line
            text_field: Field name containing the text to embed
            id_field: Field name for unique document ID
            metadata_fields: Additional fields to store as metadata
            batch_size: Number of documents to index at once
        """
        self._init_embedder()
        self._init_chroma()

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Corpus not found: {data_path}")

        # Load documents
        documents = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))

        print(f"Indexing {len(documents)} documents from {data_path}...")

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            texts = [doc[text_field] for doc in batch]
            ids = [str(doc.get(id_field, f"doc_{i+j}")) for j, doc in enumerate(batch)]

            metadatas = []
            for doc in batch:
                meta = {}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in doc:
                            # ChromaDB metadata values must be str, int, float, or bool
                            val = doc[field]
                            if isinstance(val, (str, int, float, bool)):
                                meta[field] = val
                            else:
                                meta[field] = str(val)
                metadatas.append(meta)

            # Generate embeddings
            embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

            # Add to collection
            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            print(f"  Indexed {min(i + batch_size, len(documents))}/{len(documents)}")

        print(f"✓ Indexed {len(documents)} documents into ChromaDB")

    def index_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_rows: int = 50000,
        batch_size: int = 500,
    ):
        """
        Index a HuggingFace dataset directly into ChromaDB.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "cybux/ruzivo-shona-rag")
            split: Dataset split to use
            text_column: Column containing text
            max_rows: Maximum rows to index (for RAM constraints)
            batch_size: Batch size for embedding
        """
        self._init_embedder()
        self._init_chroma()

        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets...")
            os.system("pip install datasets")
            from datasets import load_dataset

        print(f"Loading HuggingFace dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)

        if len(dataset) > max_rows:
            print(f"  Dataset has {len(dataset)} rows, using first {max_rows}")
            dataset = dataset.select(range(max_rows))

        print(f"Indexing {len(dataset)} documents...")

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            texts = batch[text_column]

            if isinstance(texts, str):
                texts = [texts]

            ids = [f"{dataset_name.replace('/', '_')}_{i+j}" for j in range(len(texts))]

            # Skip empty texts
            valid_indices = [j for j, t in enumerate(texts) if t and t.strip()]
            if not valid_indices:
                continue

            valid_texts = [texts[j] for j in valid_indices]
            valid_ids = [ids[j] for j in valid_indices]

            embeddings = self._embedder.encode(valid_texts, show_progress_bar=False).tolist()

            self._collection.add(
                ids=valid_ids,
                documents=valid_texts,
                embeddings=embeddings,
            )

            print(f"  Indexed {min(i + batch_size, len(dataset))}/{len(dataset)}")

        print(f"✓ Indexed {len(dataset)} documents from {dataset_name}")

    def query(
        self,
        question: str,
        n_results: int = 5,
        language: Optional[str] = None,
    ) -> str:
        """
        Query the RAG pipeline and return formatted context.

        Args:
            question: User's question
            n_results: Number of relevant documents to retrieve
            language: Filter by language ('shona' or 'english')

        Returns:
            Formatted string with retrieved context passages
        """
        self._init_embedder()
        self._init_chroma()

        if self._collection.count() == 0:
            return "No documents indexed. Run index_corpus() or index_huggingface_dataset() first."

        # Embed the query
        query_embedding = self._embedder.encode([question]).tolist()

        # Build filter
        where_filter = None
        if language:
            where_filter = {"language": language}

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
        )

        # Format context
        context_parts = []
        for i, (doc, dist) in enumerate(
            zip(results["documents"][0], results["distances"][0])
        ):
            relevance = max(0, 1 - dist)  # Convert cosine distance to similarity
            context_parts.append(
                f"[Source {i+1}] (relevance: {relevance:.2f})\n{doc}\n"
            )

        context = "\n---\n".join(context_parts)
        return context

    def get_stats(self) -> dict:
        """Get RAG pipeline statistics."""
        self._init_chroma()
        return {
            "collection": self.collection_name,
            "document_count": self._collection.count(),
            "persist_dir": self.persist_dir,
            "embedding_model": self.embedding_model_name,
        }


def main():
    """CLI for RAG pipeline management."""
    import argparse

    parser = argparse.ArgumentParser(description="Zim-my RAG Pipeline")
    parser.add_argument(
        "action",
        choices=["index", "query", "stats"],
        help="Action to perform",
    )
    parser.add_argument("--data", type=str, help="Path to JSONL corpus for indexing")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name for indexing")
    parser.add_argument("--question", type=str, help="Question for querying")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results")
    parser.add_argument("--max-rows", type=int, default=50000, help="Max rows from HF dataset")
    args = parser.parse_args()

    rag = ZimMyRAG()

    if args.action == "index":
        if args.data:
            rag.index_corpus(args.data)
        elif args.dataset:
            rag.index_huggingface_dataset(args.dataset, max_rows=args.max_rows)
        else:
            print("Provide --data (JSONL path) or --dataset (HuggingFace name)")
    elif args.action == "query":
        if not args.question:
            print("Provide --question for querying")
            return
        context = rag.query(args.question, n_results=args.n_results)
        print(f"\n📋 Retrieved Context:\n{context}")
    elif args.action == "stats":
        stats = rag.get_stats()
        print(f"\n📊 RAG Pipeline Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
