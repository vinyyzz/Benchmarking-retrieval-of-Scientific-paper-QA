from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import mean

import numpy as np

from .benchmark import load_benchmark
from .retrieval import CONFIGS, KS
from .runner import decode_chunk, load_query, run_directory
from .storage import VERSION, atomic_file, read_json, write_json, write_jsonl

REPORTING = {"bootstrap_seed": 23, "bootstrap_replicates": 2000, "interval": 0.95,
             "selection": ["dev Recall@5", "dev MRR@10", "B0 first on ties", "ascending config ID"],
             "evidence_length_bins": [100, 240]}


def complete_records(view, run_root, manifest_hash, split, config):
    directory = run_directory(run_root, manifest_hash, split, config)
    for question in view.questions:
        record = load_query(directory, question.question_id)
        if not record or record["status"] != "ok":
            raise ValueError(f"{split}/{config.config_id} is incomplete at {question.question_id}")
        yield record


def _metrics(row):
    values = {f"recall@{metric['k']}": metric["recall"] for metric in row["metrics"]}
    values["mrr@10"] = row["metrics"][0]["RR"]
    return values


def _means(rows):
    return {name: mean(metric[name] for metric in rows) for name in rows[0]}


def _selection_key(row):
    config_id = row["config_id"]
    return (-row["recall@5"], -row["mrr@10"], config_id != "B0", config_id)


def freeze_protocol(benchmark_root, run_root, manifest_hash):
    view = load_benchmark(benchmark_root, "dev", manifest_hash=manifest_hash)
    scores = []
    for config in CONFIGS.values():
        records = complete_records(view, run_root, manifest_hash, "dev", config)
        values = _means([_metrics(row) for row in records])
        if config.reported_k == 5:
            scores.append({"config_id": config.config_id, **values})
    ordered = sorted(scores, key=_selection_key)
    protocol = {"schema_version": VERSION, "manifest_hash": manifest_hash, "reporting": REPORTING,
                "configs": [config.record() for config in CONFIGS.values()], "dev_ranking": ordered,
                "selected": ordered[0]["config_id"],
                "qa_challenger": next(row["config_id"] for row in ordered if row["config_id"] != "B0")}
    path = Path(run_root) / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Existing frozen protocol differs from development results")
    write_json(path, protocol)
    return protocol


def paired_bootstrap(differences, paper_ids, samples=None):
    papers = sorted(set(paper_ids))
    if not papers or len(differences) != len(paper_ids):
        raise ValueError("Invalid paired bootstrap inputs")
    lookup = {paper: i for i, paper in enumerate(papers)}
    index = [lookup[paper] for paper in paper_ids]
    counts = np.bincount(index, minlength=len(papers))
    sums = np.bincount(index, weights=differences, minlength=len(papers))
    if samples is None:
        rng = np.random.default_rng(REPORTING["bootstrap_seed"])
        samples = rng.integers(len(papers), size=(REPORTING["bootstrap_replicates"], len(papers)))
    means = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _csv(path, rows):
    if rows:
        with atomic_file(path) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def analyze(benchmark_root, run_root, output, manifest_hash, split):
    output = Path(output) / split
    view = load_benchmark(benchmark_root, split, manifest_hash=manifest_hash)
    all_records = {key: list(complete_records(view, run_root, manifest_hash, split, config))
                   for key, config in CONFIGS.items()}
    baseline = [_metrics(row) for row in all_records["B0"]]
    papers = [question.paper_id for question in view.questions]
    summaries, differences, breakdowns, diagnostics = [], [], [], []
    for config_id, records in all_records.items():
        config = CONFIGS[config_id]
        values = [_metrics(row) for row in records]
        mean_values = _means(values)
        index_sizes = {row["index_artifact_id"]: row["index_bytes"] for row in records}
        summaries.append({"config_id": config_id, "chunking": config.strategy, "retrieval": config.method,
                          "embedding": config.embedding, "reported_k": config.reported_k, **mean_values,
                          "retrieval_ms_mean": mean(row["retrieval_ms"] for row in records),
                          "scoring_ms_mean": mean(row["scoring_ms"] for row in records),
                          "retrieved_tokens_mean": mean(row["retrieved_tokens"] for row in records),
                          "index_bytes": sum(index_sizes.values()), "questions": len(records)})
        for metric in mean_values:
            deltas = [row[metric] - ref[metric] for row, ref in zip(values, baseline, strict=True)]
            lower, upper = paired_bootstrap(deltas, papers)
            differences.append({"config_id": config_id, "metric": metric, "delta": mean(deltas),
                                "ci_low": lower, "ci_high": upper})
        subgroups = defaultdict(list)
        for record, metrics in zip(records, values, strict=True):
            diagnostic = record["diagnostics"][0]
            length = diagnostic["longest_gold_paragraph_tokens"]
            bucket = "≤100" if length <= 100 else "101–240" if length <= 240 else ">240"
            subgroups[("answer_type", diagnostic["answer_type"])].append(metrics)
            subgroups[("longest_gold_paragraph_tokens", bucket)].append(metrics)
        for (group, label), rows in sorted(subgroups.items()):
            breakdowns.append({"config_id": config_id, "group": group, "label": label, "questions": len(rows),
                               **_means(rows)})
        for k in KS:
            rows = [item for record in records for item in record["diagnostics"] if item["k"] == k]
            diagnostics.append({"config_id": config_id, "k": k, **{
                key: mean(row[key] for row in rows) for key in
                ("unique_gold_token_coverage", "recall_threshold_25", "recall_threshold_75")}})
    leaderboard = sorted((row for row in summaries if row["reported_k"] == 5), key=_selection_key)
    sweep = [{"config_id": row["config_id"], "k": row["reported_k"],
              "recall": row[f"recall@{row['reported_k']}"], "mrr@10": row["mrr@10"],
              "retrieved_tokens_mean": row["retrieved_tokens_mean"]}
             for row in sorted(summaries, key=lambda row: row["reported_k"])
             if row["config_id"] in {"B0", "K1", "K3", "K10"}]
    write_json(output / "summary.json", {
        "schema_version": VERSION, "manifest_hash": manifest_hash, "split": split, "papers": len(view.papers),
        "questions": len(view.questions), "exclusions": view.exclusion_counts, "leaderboard": leaderboard,
        "k_sweep": sweep, "paired_differences": differences, "breakdowns": breakdowns, "diagnostics": diagnostics,
    })
    for name, rows in (("leaderboard", leaderboard), ("k_sweep", sweep), ("paired_differences", differences),
                       ("breakdowns", breakdowns), ("diagnostics", diagnostics)):
        _csv(output / f"{name}.csv", rows)

    from .evaluator import Evaluator

    evaluator = Evaluator.open(benchmark_root, manifest_hash=manifest_hash)
    chunk_sets = {}
    traces = []
    lengths = {}
    for config_id, records in all_records.items():
        for question, row in zip(view.questions, records, strict=True):
            artifact_id = row["chunk_artifact_id"]
            if artifact_id not in chunk_sets:
                artifact = read_json(Path(run_root) / "cache" / "chunks" / f"{artifact_id}.json")
                chunks = [decode_chunk(item) for item in artifact["chunks"]]
                chunk_sets[artifact_id] = {chunk.chunk_id: chunk for chunk in chunks}
                lengths[artifact_id] = {"chunk_artifact_id": artifact_id, "paper_id": question.paper_id,
                                        "chunking": CONFIGS[config_id].strategy,
                                        "token_counts": [chunk.token_count for chunk in chunks]}
            chunks = [chunk_sets[artifact_id][item["chunk_id"]] for item in row["ranking"]]
            flags = evaluator.evidence_flags(question.question_id, chunks)
            traces.append({"config_id": config_id, "question_id": question.question_id, "question": question.question,
                           "paper_id": question.paper_id, "paper_title": view.papers[question.paper_id].title,
                           "metrics": _metrics(row), "chunks": [
                               {**rank, "text": chunk.text, "token_count": chunk.token_count, "gold_hit": hit,
                                "spans": [vars(span) for span in chunk.spans]}
                               for rank, chunk, hit in zip(row["ranking"], chunks, flags, strict=True)]})
    write_jsonl(output / "traces.jsonl", traces)
    write_jsonl(output / "chunk_lengths.jsonl", lengths.values())
    return leaderboard
