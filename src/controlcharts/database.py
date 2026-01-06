"""Vector database operations using FAISS."""

from dataclasses import dataclass, field
import numpy as np
import faiss


@dataclass
class QAPair:
    """A question-answer pair with its embedding."""
    question: str
    answer: str
    embedding: np.ndarray
    id: int = -1


@dataclass
class VectorDatabase:
    """FAISS-backed vector database for QA pairs."""

    dimension: int
    qa_pairs: list[QAPair] = field(default_factory=list)
    index: faiss.IndexFlatIP = field(init=False)

    def __post_init__(self):
        self.index = faiss.IndexFlatIP(self.dimension)

    def add(self, qa_pair: QAPair) -> None:
        """Add a QA pair to the database."""
        qa_pair.id = len(self.qa_pairs)
        self.qa_pairs.append(qa_pair)

        # Normalize and add to FAISS index
        embedding = qa_pair.embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        self.index.add(embedding)

    def add_many(self, qa_pairs: list[QAPair]) -> None:
        """Add multiple QA pairs to the database."""
        if not qa_pairs:
            return

        # Assign IDs
        start_id = len(self.qa_pairs)
        for i, qa in enumerate(qa_pairs):
            qa.id = start_id + i

        self.qa_pairs.extend(qa_pairs)

        # Batch add to FAISS
        embeddings = np.vstack([qa.embedding for qa in qa_pairs]).astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[QAPair]:
        """Search for top-k similar QA pairs."""
        if len(self.qa_pairs) == 0:
            return []

        k = min(k, len(self.qa_pairs))

        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        _, indices = self.index.search(query, k)

        return [self.qa_pairs[i] for i in indices[0] if i >= 0]

    def __len__(self) -> int:
        return len(self.qa_pairs)


def load_embeddings_cache(path: str) -> tuple[list[str], list[str], np.ndarray]:
    """Load pre-embedded QA pairs from parquet cache.

    Returns:
        Tuple of (questions, answers, embeddings)
    """
    import pandas as pd

    df = pd.read_parquet(path)
    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    embeddings = np.vstack(df["embedding"].values)

    return questions, answers, embeddings
