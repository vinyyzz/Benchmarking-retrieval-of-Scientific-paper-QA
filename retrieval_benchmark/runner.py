from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np

from .benchmark import load_benchmark
from .evaluator import Evaluator, MetricRecord, RankingError
from .models import EMBEDDINGS, Encoder, embedding_dependencies
from .retrieval import (BRANCH_DEPTH, CONFIGS, KS, RANK_LIMIT, RRF_CONSTANT, BM25Retriever,
                        Chunk, ChunkBuilder, ChunkPolicy, DenseRetriever, HybridRetriever,
                        PaperIndex, RankedChunk, Span, sparse_data)
from .storage import VERSION, atomic_file, fingerprint, read_json, read_jsonl, write_json, write_jsonl


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def run_id_for(manifest_hash, split, config):
    return fingerprint({"manifest_hash": manifest_hash, "split": split, "config": config.record()})


def run_directory(root, manifest_hash, split, config):
    return Path(root) / split / config.config_id / run_id_for(manifest_hash, split, config)


def query_path(directory, question_id):
    return directory / "queries" / f"{fingerprint(question_id)}.json"


def decode_chunk(row):
    return Chunk(row["chunk_id"], row["paper_id"], row["text"],
                 tuple(Span(**span) for span in row["spans"]), row["token_count"])


class ArtifactStore:
    def __init__(self, root, model_root, manifest, manifest_hash, evaluator):
        self.root = Path(root) / "cache"
        self.model_root = model_root
        self.manifest = manifest
        self.manifest_hash = manifest_hash
        self.evaluator = evaluator
        self.encoders = {}
        self.chunks = {}
        self.indices = {}

    def encoder(self, model):
        if model not in self.encoders:
            self.encoders[model] = Encoder(model, self.model_root)
        return self.encoders[model]

    def chunk_set(self, paper, config):
        key = (paper.paper_id, config.strategy)
        if key in self.chunks:
            return self.chunks[key]
        policy = ChunkPolicy(config.strategy)
        chunk_id = fingerprint({"manifest": self.manifest_hash, "paper_id": paper.paper_id, "policy": policy.record()})
        path = self.root / "chunks" / f"{chunk_id}.json"
        started = perf_counter()
        if path.exists():
            record = read_json(path)
            if record["chunk_artifact_id"] != chunk_id or record["paper_id"] != paper.paper_id:
                raise RankingError("Stale chunk cache")
            chunks = tuple(map(decode_chunk, record["chunks"]))
        else:
            auxiliary = self.encoder("E1") if config.strategy == "semantic" else None
            chunks = ChunkBuilder(auxiliary).build(paper, policy)
            write_json(path, {"schema_version": VERSION, "chunk_artifact_id": chunk_id, "paper_id": paper.paper_id,
                              "chunks": [asdict(chunk) for chunk in chunks]})
        self.evaluator.register_chunks(chunk_id, chunks)
        result = (chunk_id, chunks, (perf_counter() - started) * 1000)
        self.chunks[key] = result
        return result

    def index(self, paper, config):
        key = (paper.paper_id, config.strategy, config.method, config.embedding)
        if key in self.indices:
            return self.indices[key]
        chunk_id, chunks, chunk_ms = self.chunk_set(paper, config)
        sparse_id = dense_id = sparse = vectors = None
        size = 0
        started = perf_counter()
        if config.method in {"bm25", "hybrid"}:
            sparse_id = fingerprint([chunk_id, "bm25", "unicode-casefold-word", 1.5, 0.75])
            path = self.root / "sparse" / f"{sparse_id}.json"
            if not path.exists():
                write_json(path, {"index_artifact_id": sparse_id, "data": sparse_data(chunks)})
            record = read_json(path)
            if record["index_artifact_id"] != sparse_id:
                raise RankingError("Stale sparse index")
            sparse = record["data"]
            size += path.stat().st_size
        if config.embedding:
            revision = self.manifest["model_revisions"][config.embedding]
            tokenizer = self.manifest["tokenizer_revisions"][config.embedding]
            dense_id = fingerprint([chunk_id, "dense-cosine", revision, tokenizer, "no-prefix"])
            path = self.root / "dense" / f"{dense_id}.npy"
            if not path.exists():
                vectors = self.encoder(config.embedding).encode([chunk.text for chunk in chunks])
                with atomic_file(path, "wb") as stream:
                    np.save(stream, vectors, allow_pickle=False)
            else:
                vectors = np.load(path, allow_pickle=False)
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
                raise RankingError("Invalid dense index")
            size += path.stat().st_size
        index_id = fingerprint([sparse_id, dense_id, config.method, RRF_CONSTANT, BRANCH_DEPTH])
        result = (PaperIndex(paper.paper_id, self.manifest_hash, chunk_id, index_id, chunks, sparse, vectors),
                  size, chunk_ms, (perf_counter() - started) * 1000)
        self.indices[key] = result
        return result

    def rank(self, question, config, index):
        ranking_id = fingerprint([index.index_artifact_id, question.question_id, question.question,
                                  question.paper_id, "descending-score-then-chunk-id", RANK_LIMIT])
        path = self.root / "rankings" / f"{ranking_id}.json"
        if path.exists():
            row = read_json(path)
            if row["ranking_id"] != ranking_id or row["index_artifact_id"] != index.index_artifact_id:
                raise RankingError("Stale cached ranking")
            return ranking_id, tuple(RankedChunk(**item) for item in row["ranking"]), row["retrieval_ms"]
        if config.method == "bm25":
            retriever = BM25Retriever()
        elif config.method == "dense":
            retriever = DenseRetriever(self.encoder(config.embedding))
        else:
            retriever = HybridRetriever(self.encoder(config.embedding))
        started = perf_counter()
        ranking = retriever.search(question, index=index)
        latency = (perf_counter() - started) * 1000
        write_json(path, {"ranking_id": ranking_id, "index_artifact_id": index.index_artifact_id,
                          "ranking": list(map(asdict, ranking)), "retrieval_ms": latency})
        return ranking_id, ranking, latency


def load_query(directory, question_id):
    path = query_path(directory, question_id)
    if not path.exists():
        return None
    try:
        record = read_json(path)
        if (record["question_id"] != question_id or record["run_id"] != directory.name
                or record["schema_version"] != VERSION):
            return None
        if record["status"] == "ok" and (
            set(metric["k"] for metric in record["metrics"]) != set(KS)
            or len(record["metrics"]) != len(KS)
            or any(metric["question_id"] != question_id or metric["status"] != "ok"
                   or metric["run_id"] != directory.name or not isfinite(metric["recall"])
                   or not isfinite(metric["RR"]) for metric in record["metrics"])
            or len(record["diagnostics"]) != len(KS)
            or not record["ranking_id"] or not record["chunk_artifact_id"] or not record["index_artifact_id"]
            or len(record["ranking"]) > RANK_LIMIT
            or any(row["rank"] != rank or not isfinite(row["score"])
                   for rank, row in enumerate(record["ranking"], 1))
            or len({row["chunk_id"] for row in record["ranking"]}) != len(record["ranking"])
        ):
            return None
        return record
    except (ValueError, TypeError, KeyError, OSError):
        return None


def _evaluate_one(store, evaluator, paper, question, config, run_id, attempt):
    record = {"schema_version": VERSION, "run_id": run_id, "question_id": question.question_id,
              "paper_id": question.paper_id, "attempt": attempt, "ranking_id": None, "ranking": [],
              "chunk_artifact_id": None, "index_artifact_id": None, "retrieval_ms": None,
              "latency_ms": None, "scoring_ms": None, "retrieved_tokens": None,
              "tokens_by_k": {}, "diagnostics": [], "index_bytes": None, "chunk_count": None}
    phase = "retrieval_error"
    try:
        index, size, chunk_ms, index_ms = store.index(paper, config)
        record.update(chunk_artifact_id=index.chunk_artifact_id, index_artifact_id=index.index_artifact_id,
                      index_bytes=size, chunk_count=len(index.chunks), chunk_preparation_ms=chunk_ms,
                      index_preparation_ms=index_ms)
        rank_id, ranking, latency = store.rank(question, config, index)
        record.update(ranking_id=rank_id, ranking=list(map(asdict, ranking)), retrieval_ms=latency, latency_ms=latency)
        phase = "evaluation_error"
        started = perf_counter()
        metrics = evaluator.score_query(run_id, question.question_id, ranking, chunk_artifact_id=index.chunk_artifact_id)
        diagnostics = list(evaluator.diagnostics(question.question_id, ranking,
                                                 chunk_artifact_id=index.chunk_artifact_id))
        counts = {chunk.chunk_id: chunk.token_count for chunk in index.chunks}
        tokens = {str(k): sum(counts[row.chunk_id] for row in ranking[:k]) for k in KS}
        record.update(status="ok", error_code=None, metrics=list(map(asdict, metrics)), diagnostics=diagnostics,
                      scoring_ms=(perf_counter() - started) * 1000, tokens_by_k=tokens,
                      retrieved_tokens=tokens[str(config.reported_k)])
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        status = "validation_error" if isinstance(exc, RankingError) else phase
        error_code = type(exc).__name__
        record.update(status=status, error_code=error_code, error=str(exc), metrics=[
            asdict(MetricRecord(VERSION, run_id, question.question_id, k, None, None, 0, status, error_code))
            for k in KS])
    return record


def export_run(directory, records):
    write_jsonl(directory / "rankings.jsonl", (
        {"run_id": row["run_id"], "ranking_id": row["ranking_id"], "question_id": row["question_id"], **rank}
        for row in records for rank in row["ranking"]))
    write_jsonl(directory / "metrics.jsonl", (metric for row in records for metric in row["metrics"]))
    write_jsonl(directory / "diagnostics.jsonl", (item for row in records for item in row["diagnostics"]))
    write_jsonl(directory / "query_status.jsonl", (
        {key: value for key, value in row.items() if key not in {"ranking", "metrics", "diagnostics"}}
        for row in records))


def run_study(benchmark_root, run_root, model_root, manifest_hash, split, config_ids, limit=None, budget=None):
    if limit is not None and (split != "dev" or limit < 1):
        raise ValueError("Pilot limits are supported only on the development split")
    if split == "heldout":
        protocol = read_json(Path(run_root) / "protocol.json")
        if (protocol["manifest_hash"] != manifest_hash
                or protocol["configs"] != [config.record() for config in CONFIGS.values()]):
            raise ValueError("Held-out protocol does not match the frozen benchmark and suite")
    view = load_benchmark(benchmark_root, split, manifest_hash=manifest_hash)
    revisions = {key: value[1] for key, value in EMBEDDINGS.items()}
    if (view.manifest["model_revisions"] != revisions or view.manifest["tokenizer_revisions"] != revisions
            or view.manifest["dependencies"] != embedding_dependencies()):
        raise ValueError("Runtime revisions differ from the benchmark manifest")
    evaluator = Evaluator.open(benchmark_root, manifest_hash=manifest_hash)
    store = ArtifactStore(run_root, model_root, view.manifest, manifest_hash, evaluator)
    success = True
    for config_id in config_ids:
        config = CONFIGS[config_id]
        directory = run_directory(run_root, manifest_hash, split, config)
        run_path = directory / "run.json"
        started_at = read_json(run_path)["started_at"] if run_path.exists() else timestamp()
        selected = view.questions if limit is None else view.questions[:limit]
        paused = False
        for question in selected:
            previous = load_query(directory, question.question_id)
            if previous and previous["status"] == "ok":
                continue
            if budget is not None and not budget.take():
                paused = True
                break
            attempt = 1 if previous is None else previous["attempt"] + 1
            record = _evaluate_one(store, evaluator, view.papers[question.paper_id], question, config,
                                   directory.name, attempt)
            write_json(query_path(directory, question.question_id), record)
            if record["status"] != "ok":
                print(f"{split} {config_id} {question.question_id}: {record['status']} {record['error']}", flush=True)
        records = [row for question in view.questions if (row := load_query(directory, question.question_id))]
        completed = sum(row["status"] == "ok" for row in records)
        pending = len(view.questions) - len(records)
        errors = len(records) - completed
        final_status = "complete" if completed == len(view.questions) else "incomplete"
        write_json(run_path, {"schema_version": VERSION, "run_id": directory.name, "config_id": config_id,
                              "manifest_hash": manifest_hash, "split": split, "started_at": started_at,
                              "completed_at": timestamp() if final_status == "complete" else None,
                              "final_status": final_status, "ok": completed, "pending": pending, "errors": errors,
                              "config": config.record(), "model_revisions": revisions, "evaluator_version": VERSION,
                              "chunk_artifact_ids": sorted({row["chunk_artifact_id"] for row in records
                                                            if row["chunk_artifact_id"]}),
                              "index_artifact_ids": sorted({row["index_artifact_id"] for row in records
                                                            if row["index_artifact_id"]})})
        export_run(directory, records)
        print(f"{split} {config_id}: {completed}/{len(view.questions)} successful, {pending} pending, {errors} errors",
              flush=True)
        success &= errors == 0 and (limit is not None or paused or final_status == "complete")
        if paused:
            print("Retrieval batch ended at a checkpoint. Rerun the same command to continue.", flush=True)
            break
    return success


def retrieval_progress(benchmark_root, run_root, manifest_hash):
    questions = {"dev": [], "heldout": []}
    for row in read_jsonl(Path(benchmark_root) / "questions.jsonl"):
        questions[row["split"]].append(row["question_id"])
    for split, identifiers in questions.items():
        ok = errors = 0
        for config in CONFIGS.values():
            directory = run_directory(run_root, manifest_hash, split, config)
            for question_id in identifiers:
                record = load_query(directory, question_id)
                if record:
                    ok += record["status"] == "ok"
                    errors += record["status"] != "ok"
        total = len(identifiers) * len(CONFIGS)
        yield split, ok, errors, total
