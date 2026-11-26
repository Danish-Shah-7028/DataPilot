import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import faiss
import numpy as np

from datapilot.data.schema import ColumnProfile, SchemaProfile, TableProfile

TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; splits snake_case and camelCase and drops a plural 's'."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ").lower()
    tokens = TOKEN_PATTERN.findall(spaced)
    return [token[:-1] if len(token) > 3 and token.endswith("s") else token for token in tokens]


def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


@dataclass(frozen=True)
class HashingEmbedder:
    """Bag-of-words embedder with no model download; used in tests and as an offline fallback."""

    dimensions: int = 512

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions
                vectors[row, bucket] += 1.0
        return normalise_rows(vectors)


@lru_cache(maxsize=2)
def load_sentence_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


@dataclass(frozen=True)
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = load_sentence_model(self.model_name)
        vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


def embedder_for(model_name: str) -> Embedder:
    """`hashing` picks the lightweight embedder, which keeps PyTorch out of memory on
    small hosts; any other value is loaded as a sentence-transformers model."""
    if model_name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)


@dataclass(frozen=True)
class SchemaDocument:
    table: str
    column: str | None
    text: str


def humanise(name: str) -> str:
    return " ".join(tokenize(name)) or name


def table_document(table: TableProfile) -> SchemaDocument:
    columns = ", ".join(humanise(column.name) for column in table.columns)
    return SchemaDocument(table.name, None, f"table {humanise(table.name)} with {columns}")


def column_document(table: TableProfile, column: ColumnProfile) -> SchemaDocument:
    text = f"{humanise(column.name)} of {humanise(table.name)} ({column.dtype.lower()})"
    if column.original_name:
        text += f", originally {column.original_name}"
    if column.sample_values:
        text += ", for example " + ", ".join(column.sample_values[:3])
    return SchemaDocument(table.name, column.name, text)


@dataclass
class SchemaIndex:
    """FAISS inner-product index over one document per table and one per column."""

    profile: SchemaProfile
    documents: list[SchemaDocument]
    index: Any
    embedder: Embedder

    def search(self, query: str, k: int | None = None) -> list[tuple[SchemaDocument, float]]:
        k = min(k or len(self.documents), len(self.documents))
        if k == 0:
            return []
        scores, ids = self.index.search(self.embedder.encode([query]), k)
        return [
            (self.documents[doc_id], float(score))
            for score, doc_id in zip(scores[0], ids[0], strict=True)
            if doc_id >= 0
        ]


def build_schema_index(profile: SchemaProfile, embedder: Embedder) -> SchemaIndex:
    documents = []
    for table in profile.tables:
        documents.append(table_document(table))
        documents += [column_document(table, column) for column in table.columns]
    vectors = embedder.encode([document.text for document in documents])
    index = faiss.IndexFlatIP(vectors.shape[1])
    if len(documents):
        index.add(vectors)
    return SchemaIndex(profile, documents, index, embedder)
