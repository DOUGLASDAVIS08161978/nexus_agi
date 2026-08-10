import os
import json
import uuid
import sqlite3
import datetime
from typing import List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _serialize_embedding(emb: np.ndarray) -> bytes:
    return emb.tobytes()


def _deserialize_embedding(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class ContextualMemory:
    """
    A lightweight contextual memory system that stores interactions,
    indexes them with vector embeddings, and retrieves relevant moments.
    """

    def __init__(
        self,
        db_path: str = "lumina_memory.db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: Optional[int] = None,
    ):
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        if self.embedding_dim is None:
            self.embedding_dim = SentenceTransformer(self.embedding_model_name).get_sentence_embedding_dimension()
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._create_tables()

    def _create_tables(self) -> None:
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                speaker TEXT,
                content TEXT,
                embedding BLOB
            )
            """
        )
        self.conn.commit()

    def _embed_text(self, text: str) -> np.ndarray:
        model = SentenceTransformer(self.embedding_model_name)
        return model.encode(text, convert_to_tensor=True)

    def add_interaction(self, speaker: str, content: str) -> None:
        embedding = self._embed_text(content)
        self.cur.execute(
            """
            INSERT INTO interactions (id, timestamp, speaker, content, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), datetime.datetime.now().isoformat(), speaker, content, _serialize_embedding(embedding))
        )
        self.conn.commit()

    def retrieve(self, speaker: str, num_results: int = 10) -> List[Tuple[str, str, np.ndarray]]:
        self.cur.execute(
            """
            SELECT content, embedding FROM interactions WHERE speaker = ? ORDER BY timestamp DESC LIMIT ?
            """,
            (speaker, num_results)
        )
        return [(row[0], _deserialize_embedding(row[1], self.embedding_dim)) for row in self.cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
