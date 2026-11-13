from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass
from math import log1p
import re
from typing import Protocol, Sequence

import numpy as np

from .benchmark import Paper, Question
from .storage import VERSION, fingerprint

KS = (1, 3, 5, 10)
TARGET_TOKENS = 200
MAX_TOKENS = 240
BOUNDARY_COSINE = 0.70
BM25_K1 = 1.5
BM25_B = 0.75
RRF_CONSTANT = 60
BRANCH_DEPTH = 50
RANK_LIMIT = 10


@dataclass(frozen=True)
class Span:
    paper_id: str
    paragraph_id: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    spans: tuple[Span, ...]
    token_count: int


@dataclass(frozen=True)
class RankedChunk:
    rank: int
    chunk_id: str
    score: float


@dataclass(frozen=True)
class ChunkPolicy:
    strategy: str

    def record(self):
        policy = {"strategy": self.strategy, "target_tokens": TARGET_TOKENS,
                  "max_tokens": MAX_TOKENS, "overlap_tokens": 0}
        if self.strategy == "semantic":
            policy.update(auxiliary_encoder="E1", boundary_cosine_threshold=BOUNDARY_COSINE)
        return policy


@dataclass(frozen=True)
class Config:
    config_id: str
    strategy: str
    method: str
    embedding: str | None
    reported_k: int = 5

    def record(self):
        retrieval = {"method": self.method, "embedding": self.embedding, "ranking_limit": RANK_LIMIT}
        if self.embedding:
            retrieval["exact_search"] = True
        if self.method in {"bm25", "hybrid"}:
            retrieval.update(k1=BM25_K1, b=BM25_B)
        if self.method == "hybrid":
            retrieval.update(rrf_constant=RRF_CONSTANT, branch_depth=BRANCH_DEPTH)
        return {"schema_version": VERSION, "config_id": self.config_id, "scope": "source_paper",
                "chunking": ChunkPolicy(self.strategy).record(), "retrieval": retrieval,
                "evaluation": {"reported_k": self.reported_k, "ks": list(KS),
                               "paragraph_threshold": 0.5, "mrr_cutoff": 10}}


CONFIGS = {config.config_id: config for config in (
    Config("B0", "fixed", "dense", "E1"),
    Config("C1", "semantic", "dense", "E1"),
    Config("C2", "section_aware", "dense", "E1"),
    Config("R1", "fixed", "bm25", None),
    Config("R2", "fixed", "hybrid", "E1"),
    Config("E2", "fixed", "dense", "E2"),
    Config("K1", "fixed", "dense", "E1", 1),
    Config("K3", "fixed", "dense", "E1", 3),
    Config("K10", "fixed", "dense", "E1", 10),
    Config("I1", "semantic", "bm25", None),
    Config("I2", "semantic", "hybrid", "E1"),
    Config("I3", "section_aware", "bm25", None),
    Config("I4", "section_aware", "hybrid", "E1"),
)}


def _span(paragraph, start, end):
    offsets = paragraph.token_offsets[start:end]
    return Span(paragraph.paper_id, paragraph.paragraph_id, min(pair[0] for pair in offsets),
                max(pair[1] for pair in offsets), start, end)


def _count(span):
    return span.token_end - span.token_start


def _sentence_spans(paragraph):
    starts = [start for start, _ in paragraph.token_offsets]
    boundaries = [0]
    for match in re.finditer(r"(?<=[.!?])\s+(?=\S)", paragraph.text):
        following = paragraph.text[match.end()]
        if following.isupper() or following.isdigit():
            boundaries.append(bisect_left(starts, match.end()))
    boundaries.append(len(starts))
    for start, end in zip(boundaries, boundaries[1:]):
        for offset in range(start, end, MAX_TOKENS):
            yield _span(paragraph, offset, min(offset + MAX_TOKENS, end))


def _fixed_windows(paragraphs):
    pending, size = [], 0
    for paragraph in paragraphs:
        start = 0
        while start < paragraph.canonical_token_count:
            end = min(start + TARGET_TOKENS - size, paragraph.canonical_token_count)
            pending.append(_span(paragraph, start, end))
            size += end - start
            start = end
            if size == TARGET_TOKENS:
                yield pending
                pending, size = [], 0
    if pending:
        yield pending


def _packed_windows(units, breaks):
    pending, size = [], 0
    for unit, boundary in zip(units, breaks, strict=True):
        length = _count(unit)
        if pending and (boundary or size + length > MAX_TOKENS):
            yield pending
            pending, size = [], 0
        pending.append(unit)
        size += length
        if size >= TARGET_TOKENS:
            yield pending
            pending, size = [], 0
    if pending:
        yield pending


def _merge_spans(spans):
    merged = []
    for span in spans:
        if merged and merged[-1].paragraph_id == span.paragraph_id and merged[-1].token_end == span.token_start:
            last = merged.pop()
            span = Span(span.paper_id, span.paragraph_id, min(last.char_start, span.char_start),
                        max(last.char_end, span.char_end), last.token_start, span.token_end)
        merged.append(span)
    return tuple(merged)


class ChunkBuilder:
    def __init__(self, auxiliary_encoder=None):
        self.auxiliary_encoder = auxiliary_encoder

    def build(self, paper: Paper, policy: ChunkPolicy) -> Sequence[Chunk]:
        paragraphs = {paragraph.paragraph_id: paragraph for paragraph in paper.paragraphs}

        def render(span):
            return paragraphs[span.paragraph_id].text[span.char_start:span.char_end]

        if policy.strategy == "fixed":
            windows = _fixed_windows(paper.paragraphs)
        else:
            units = [unit for paragraph in paper.paragraphs for unit in _sentence_spans(paragraph)]
            if not units:
                return ()
            if policy.strategy == "section_aware":
                groups = [paragraphs[span.paragraph_id].section_id for span in units]
                breaks = [left != right for left, right in zip(groups, groups[1:])]
            elif policy.strategy == "semantic" and self.auxiliary_encoder is not None:
                vectors = self.auxiliary_encoder.encode([render(unit) for unit in units])
                similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
                breaks = [similarity < BOUNDARY_COSINE for similarity in similarities]
            else:
                raise ValueError(f"Unsupported or unconfigured chunk strategy {policy.strategy}")
            windows = _packed_windows(units, [False, *breaks])

        chunks = []
        for window in windows:
            spans = _merge_spans(window)
            chunk_id = fingerprint([paper.paper_id, policy.record(), [asdict(span) for span in spans]])
            chunks.append(Chunk(chunk_id, paper.paper_id, "\n".join(map(render, spans)),
                                spans, sum(map(_count, spans))))
        return tuple(chunks)


def bm25_tokens(text):
    return re.findall(r"\w+", text.casefold())


def sparse_data(chunks):
    counts = [Counter(bm25_tokens(chunk.text)) for chunk in chunks]
    lengths = [sum(row.values()) for row in counts]
    frequencies = Counter(term for row in counts for term in row)
    return {"counts": counts, "lengths": lengths, "df": frequencies}


@dataclass
class PaperIndex:
    paper_id: str
    manifest_hash: str
    chunk_artifact_id: str
    index_artifact_id: str
    chunks: Sequence[Chunk]
    sparse: dict | None = None
    vectors: np.ndarray | None = None


class Retriever(Protocol):
    def search(self, query: Question, *, index: PaperIndex, limit: int = 10) -> Sequence[RankedChunk]: ...


def _validate_search(query, index, limit):
    if query.paper_id != index.paper_id:
        raise ValueError("Question and index refer to different papers")
    if not 1 <= limit <= BRANCH_DEPTH:
        raise ValueError("Invalid ranking limit")


def _rank(chunks, scores, limit):
    ordered = sorted(zip(chunks, scores, strict=True), key=lambda pair: (-pair[1], pair[0].chunk_id))
    return tuple(RankedChunk(rank, chunk.chunk_id, float(score))
                 for rank, (chunk, score) in enumerate(ordered[:limit], 1))


class BM25Retriever:
    def search(self, query: Question, *, index: PaperIndex, limit: int = 10) -> Sequence[RankedChunk]:
        _validate_search(query, index, limit)
        if not index.chunks:
            return ()
        data = index.sparse
        n = len(index.chunks)
        average = sum(data["lengths"]) / n
        if not average:
            return _rank(index.chunks, [0.0] * n, limit)
        terms = sorted(set(bm25_tokens(query.question)) & data["df"].keys())
        idf = {term: log1p((n - data["df"][term] + 0.5) / (data["df"][term] + 0.5)) for term in terms}
        scores = []
        for counts, length in zip(data["counts"], data["lengths"], strict=True):
            penalty = BM25_K1 * (1 - BM25_B + BM25_B * length / average)
            scores.append(sum(idf[term] * counts[term] * (BM25_K1 + 1) / (counts[term] + penalty)
                              for term in terms if counts.get(term, 0)))
        return _rank(index.chunks, scores, limit)


class DenseRetriever:
    def __init__(self, encoder):
        self.encoder = encoder

    def search(self, query: Question, *, index: PaperIndex, limit: int = 10) -> Sequence[RankedChunk]:
        _validate_search(query, index, limit)
        if not index.chunks:
            return ()
        scores = index.vectors @ self.encoder.encode([query.question])[0]
        return _rank(index.chunks, scores, limit)


class HybridRetriever:
    def __init__(self, encoder):
        self.dense = DenseRetriever(encoder)
        self.sparse = BM25Retriever()

    def search(self, query: Question, *, index: PaperIndex, limit: int = 10) -> Sequence[RankedChunk]:
        _validate_search(query, index, limit)
        scores = Counter()
        for branch in (self.sparse, self.dense):
            for row in branch.search(query, index=index, limit=BRANCH_DEPTH):
                scores[row.chunk_id] += 1 / (RRF_CONSTANT + row.rank)
        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return tuple(RankedChunk(rank, chunk_id, score)
                     for rank, (chunk_id, score) in enumerate(ordered[:limit], 1))
