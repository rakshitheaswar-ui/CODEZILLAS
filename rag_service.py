#!/usr/bin/env python3
"""
Lightweight RAG over local project files using TF-IDF with cosine similarity.
Supports building an index and querying top-k relevant chunks.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


class SimpleRAG:
    """Simple TF-IDF based RAG for local files."""

    def __init__(self, root_dir: str, include_exts: Tuple[str, ...] = (".py", ".md", ".txt")):
        self.root_dir = root_dir
        self.include_exts = include_exts
        self.documents: List[str] = []
        self.doc_paths: List[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.cache_dir: str | None = None

    def build(self) -> "SimpleRAG":
        texts: List[str] = []
        paths: List[str] = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                if fn.startswith('.'):
                    continue
                fp = os.path.join(dirpath, fn)
                if not fp.lower().endswith(self.include_exts):
                    continue
                text = _read_text(fp)
                if not text:
                    continue
                # Lightweight cleaning; strip long whitespace
                text = re.sub(r"\s+", " ", text)
                texts.append(text)
                paths.append(fp)

        self.documents = texts
        self.doc_paths = paths
        if not self.documents:
            self.vectorizer = None
            self.matrix = None
            return self

        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
        self.matrix = self.vectorizer.fit_transform(self.documents)
        return self

    def save(self, cache_dir: str) -> None:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            joblib.dump({
                'vectorizer': self.vectorizer,
                'matrix': self.matrix,
                'documents': self.documents,
                'doc_paths': self.doc_paths
            }, os.path.join(cache_dir, 'rag_index.joblib'))
            self.cache_dir = cache_dir
        except Exception:
            pass

    def load(self, cache_dir: str) -> bool:
        try:
            data = joblib.load(os.path.join(cache_dir, 'rag_index.joblib'))
            self.vectorizer = data.get('vectorizer')
            self.matrix = data.get('matrix')
            self.documents = data.get('documents') or []
            self.doc_paths = data.get('doc_paths') or []
            self.cache_dir = cache_dir
            return self.is_ready()
        except Exception:
            return False

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.matrix is not None and len(self.documents) > 0

    def query(self, question: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if not self.is_ready():
            return []
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        idxs = np.argsort(-sims)[:top_k]
        results: List[Tuple[str, str, float]] = []
        for i in idxs:
            results.append((self.doc_paths[i], self.documents[i][:2000], float(sims[i])))
        return results


