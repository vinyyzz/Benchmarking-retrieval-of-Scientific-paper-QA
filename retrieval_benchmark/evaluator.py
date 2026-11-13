from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from statistics import mean
from typing import Sequence

from .benchmark import PRIVATE_FILES, verified_manifest
from .retrieval import KS, MAX_TOKENS, RANK_LIMIT, Chunk, RankedChunk
from .storage import VERSION, read_jsonl


class RankingError(ValueError):
    """A ranking has invalid data or provenance."""


@dataclass(frozen=True)
class MetricRecord:
    schema_version: str
    run_id: str
    question_id: str
    k: int
    recall: float | None
    RR: float | None
    annotation_count: int
    status: str
    error_code: str | None


def _union_length(intervals):
    covered, end = 0, 0
    for start, stop in sorted(intervals):
        covered += max(0, stop - max(start, end))
        end = max(end, stop)
    return covered


def _intervals(chunks):
    result = defaultdict(list)
    for chunk in chunks:
        for span in chunk.spans:
            result[span.paragraph_id].append((span.token_start, span.token_end))
    return result


class Evaluator:
    def __init__(self, questions, paragraphs, gold, references):
        self._questions = questions
        self._paragraphs = paragraphs
        self._gold = gold
        self._references = references
        self._artifacts = {}

    @classmethod
    def open(cls, root: Path, *, manifest_hash: str, api_version: str = "1.0.0") -> "Evaluator":
        root = Path(root)
        if api_version != VERSION:
            raise ValueError("Unsupported evaluator API")
        verified_manifest(root, manifest_hash, ("questions.jsonl", "paragraphs.jsonl", *PRIVATE_FILES))
        questions = {row["question_id"]: row["paper_id"] for row in read_jsonl(root / "questions.jsonl")}
        paragraphs = {row["paragraph_id"]: row for row in read_jsonl(root / "paragraphs.jsonl")}
        gold, references, seen = defaultdict(list), defaultdict(list), set()
        for row in read_jsonl(root / "gold.jsonl"):
            question_id = row["question_id"]
            ids = set(row["paragraph_ids"])
            key = (question_id, row["annotation_id"])
            if (question_id not in questions or not ids or key in seen
                    or any(pid not in paragraphs or paragraphs[pid]["paper_id"] != questions[question_id]
                           or not paragraphs[pid]["canonical_token_count"] for pid in ids)):
                raise ValueError(f"Invalid gold annotation {key}")
            seen.add(key)
            gold[question_id].append(frozenset(ids))
        reference_keys = set()
        for row in read_jsonl(root / "references.jsonl"):
            key = (row["question_id"], row["annotation_id"])
            if (key not in seen or key in reference_keys or not row["answer"].strip()
                    or row["answer_type"] not in {"extractive", "abstractive", "boolean"}):
                raise ValueError(f"Invalid reference {key}")
            reference_keys.add(key)
            references[row["question_id"]].append((row["answer_type"], row["answer"]))
        if set(gold) != set(questions) or reference_keys != seen:
            raise ValueError("Missing gold evidence or reference answers")
        return cls(questions, paragraphs, gold, references)

    def register_chunks(self, chunk_artifact_id: str, chunks: Sequence[Chunk]):
        mapped = {}
        for chunk in chunks:
            if chunk.chunk_id in mapped or not 0 < chunk.token_count <= MAX_TOKENS:
                raise RankingError("Duplicate chunk or invalid token count")
            rendered = []
            for span in chunk.spans:
                paragraph = self._paragraphs.get(span.paragraph_id)
                if (paragraph is None or span.paper_id != chunk.paper_id or span.paper_id != paragraph["paper_id"]
                        or not 0 <= span.char_start < span.char_end <= len(paragraph["text"])
                        or not 0 <= span.token_start < span.token_end <= paragraph["canonical_token_count"]):
                    raise RankingError(f"Invalid span in {chunk.chunk_id}")
                offsets = paragraph["token_offsets"][span.token_start:span.token_end]
                if span.char_start != min(pair[0] for pair in offsets) or span.char_end != max(pair[1] for pair in offsets):
                    raise RankingError(f"Character offsets do not match tokens in {chunk.chunk_id}")
                rendered.append(paragraph["text"][span.char_start:span.char_end])
            intervals = _intervals([chunk])
            if (sum(map(_union_length, intervals.values())) != chunk.token_count
                    or sum(span.token_end - span.token_start for span in chunk.spans) != chunk.token_count
                    or "\n".join(rendered) != chunk.text):
                raise RankingError(f"Invalid chunk provenance for {chunk.chunk_id}")
            mapped[chunk.chunk_id] = chunk
        if chunk_artifact_id in self._artifacts and self._artifacts[chunk_artifact_id] != mapped:
            raise RankingError("Chunk artifact ID already registered with different contents")
        self._artifacts[chunk_artifact_id] = mapped

    def _resolve(self, question_id, ranking, chunk_artifact_id):
        if question_id not in self._gold:
            raise ValueError(f"Missing gold for {question_id}")
        if chunk_artifact_id not in self._artifacts:
            raise RankingError("Unknown chunk artifact")
        chunks = self._artifacts[chunk_artifact_id]
        if len(ranking) > RANK_LIMIT or len({row.chunk_id for row in ranking}) != len(ranking):
            raise RankingError("Too many or duplicate ranked chunks")
        resolved = []
        for rank, row in enumerate(ranking, 1):
            chunk = chunks.get(row.chunk_id)
            if (type(row.rank) is not int or row.rank != rank or not isinstance(row.score, Real)
                    or isinstance(row.score, bool) or not isfinite(row.score) or chunk is None
                    or chunk.paper_id != self._questions[question_id]):
                raise RankingError(f"Invalid rank or chunk provenance at rank {rank}")
            resolved.append(chunk)
        return resolved

    def _mapped(self, chunk, threshold):
        return {pid for pid, spans in _intervals([chunk]).items()
                if _union_length(spans) / self._paragraphs[pid]["canonical_token_count"] >= threshold}

    def _recall(self, mapped, question_id):
        return mean(len(mapped & annotation) / len(annotation) for annotation in self._gold[question_id])

    def score_query(self, run_id: str, question_id: str, ranking: Sequence[RankedChunk], *,
                    chunk_artifact_id: str, ks: tuple[int, ...] = KS) -> Sequence[MetricRecord]:
        if not ks or len(set(ks)) != len(ks) or any(k not in KS for k in ks):
            raise ValueError("Invalid metric cutoffs")
        chunks = self._resolve(question_id, ranking, chunk_artifact_id)
        mapped = [self._mapped(chunk, 0.5) for chunk in chunks]
        annotations = self._gold[question_id]
        rr = mean(next((1 / rank for rank, ids in enumerate(mapped, 1) if ids & gold), 0.0)
                  for gold in annotations)
        return tuple(MetricRecord(VERSION, run_id, question_id, k,
                                  self._recall(set().union(*mapped[:k]), question_id),
                                  rr, len(annotations), "ok", None) for k in ks)

    def diagnostics(self, question_id, ranking, *, chunk_artifact_id):
        chunks = self._resolve(question_id, ranking, chunk_artifact_id)
        annotations = self._gold[question_id]
        lengths = {pid: self._paragraphs[pid]["canonical_token_count"] for pid in set().union(*annotations)}
        mapped = {threshold: [self._mapped(chunk, threshold) for chunk in chunks] for threshold in (0.25, 0.75)}
        for k in KS:
            coverage = {pid: _union_length(spans) for pid, spans in _intervals(chunks[:k]).items()}
            yield {
                "question_id": question_id, "k": k,
                "unique_gold_token_coverage": mean(sum(coverage.get(pid, 0) for pid in gold) /
                                                   sum(lengths[pid] for pid in gold) for gold in annotations),
                "recall_threshold_25": self._recall(set().union(*mapped[0.25][:k]), question_id),
                "recall_threshold_75": self._recall(set().union(*mapped[0.75][:k]), question_id),
                "longest_gold_paragraph_tokens": max(lengths.values()),
                "answer_type": self.answer_type(question_id),
            }

    def answer_type(self, question_id):
        kinds = {kind for kind, _ in self._references[question_id]}
        return next(kind for kind in ("extractive", "abstractive", "boolean") if kind in kinds)

    def extractive_references(self, question_id):
        return tuple(text for kind, text in self._references[question_id] if kind == "extractive")

    def gold_hit(self, question_id, chunks):
        gold = set().union(*self._gold[question_id])
        return any(self._mapped(chunk, 0.5) & gold for chunk in chunks)

    def evidence_flags(self, question_id, chunks):
        gold = set().union(*self._gold[question_id])
        return [bool(self._mapped(chunk, 0.5) & gold) for chunk in chunks]
