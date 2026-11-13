from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .analysis import complete_records
from .benchmark import load_benchmark
from .retrieval import CONFIGS
from .runner import run_directory
from .storage import VERSION, atomic_file, file_hash, read_json, read_jsonl


def retrieval_report(dev, heldout, protocol):
    baseline = next(row for row in heldout["leaderboard"] if row["config_id"] == "B0")
    chosen_id = protocol["selected"]
    chosen = next(row for row in heldout["leaderboard"] if row["config_id"] == chosen_id)
    delta = next(row for row in heldout["paired_differences"]
                 if row["config_id"] == chosen_id and row["metric"] == "recall@5")
    rows = [
        "# Benchmarking Retrieval for Scientific-Paper QA", "",
        "## Retrieval study", "",
        f"The frozen benchmark contains {dev['papers']} development and {heldout['papers']} held-out papers, "
        f"with {dev['questions']} and {heldout['questions']} eligible questions respectively. "
        "Each question searches its associated paper. Recall measures coverage of text evidence paragraphs.",
        "", f"Development selection chose **{chosen_id}**. On held-out papers it scored "
        f"{chosen['recall@5']:.3f} Recall@5 and {chosen['mrr@10']:.3f} MRR@10. "
        f"B0 scored {baseline['recall@5']:.3f} and {baseline['mrr@10']:.3f}. "
        f"The paired Recall@5 change was {delta['delta']:+.3f} "
        f"(95% paper-cluster interval {delta['ci_low']:+.3f} to {delta['ci_high']:+.3f}).", "",
        "| Setting | Chunking | Retrieval | Embedding | Recall@5 | MRR@10 | Mean context tokens |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    rows.extend(f"| {r['config_id']} | {r['chunking']} | {r['retrieval']} | {r['embedding'] or '—'} | "
                f"{r['recall@5']:.3f} | {r['mrr@10']:.3f} | {r['retrieved_tokens_mean']:.1f} |"
                for r in heldout["leaderboard"])
    rows.extend(["", "### Configuration guidance", ""])
    if delta["ci_low"] > 0:
        rows.append(f"Use {chosen_id} when evidence recall is the priority. Its paired Recall@5 gain over B0 "
                    "was positive across the reported interval. Compare index size and latency in the exported cost table.")
    else:
        rows.append("B0 remains a reasonable default for this benchmark because the development-selected setting "
                    "did not show a clear paired Recall@5 improvement. Use the cost table to compare cheaper settings.")
    fastest = min(heldout["leaderboard"], key=lambda row: row["retrieval_ms_mean"])
    rows.append(f"For a latency-sensitive run, {fastest['config_id']} had the lowest measured mean query latency "
                f"at {fastest['retrieval_ms_mean']:.1f} ms, with Recall@5 of {fastest['recall@5']:.3f}. "
                "Latencies come from one execution environment and exclude index construction.")
    semantic = next(row for row in heldout["leaderboard"] if row["config_id"] == "C1")
    rows.extend(["", f"The semantic boundary rule at cosine 0.70 produced much shorter contexts. At the same "
                 f"k=5, C1 retrieved {semantic['retrieved_tokens_mean']:.1f} canonical tokens per question "
                 f"and achieved Recall@5 of {semantic['recall@5']:.3f}. B0 retrieved "
                 f"{baseline['retrieved_tokens_mean']:.1f} tokens per question. The comparison holds chunk count fixed."])
    rows.extend(["", "The k sweep uses prefixes of the same B0 ranking."])
    rows.extend(f"- k={r['k']} recovered {r['recall']:.3f} recall with "
                f"{r['retrieved_tokens_mean']:.1f} mean canonical tokens." for r in heldout["k_sweep"])
    return rows


def write_retrieval_report(output, run_root):
    output = Path(output)
    rows = retrieval_report(read_json(output / "dev" / "summary.json"),
                            read_json(output / "heldout" / "summary.json"),
                            read_json(Path(run_root) / "protocol.json"))
    with atomic_file(output / "retrieval_report.md") as stream:
        stream.write("\n".join(rows) + "\n")


def package_release(benchmark_root, runs, output, destination, manifest_hash):
    output, runs, destination = Path(output), Path(runs), Path(destination)
    dev, heldout = (read_json(output / split / "summary.json") for split in ("dev", "heldout"))
    protocol = read_json(runs / "protocol.json")
    if any(row["manifest_hash"] != manifest_hash for row in (dev, heldout, protocol)):
        raise ValueError("Results and protocol must match the benchmark manifest")
    for split in ("dev", "heldout"):
        view = load_benchmark(benchmark_root, split, manifest_hash=manifest_hash)
        for config in CONFIGS.values():
            list(complete_records(view, runs, manifest_hash, split, config))
    qa_dir = output / "heldout"
    qa = read_json(qa_dir / "qa_summary.json")
    agreement = read_json(qa_dir / "judge_agreement.json")
    qa_protocol = read_json(runs / "qa_protocol.json")
    configs = ["B0", protocol["qa_challenger"]]
    answers = list(read_jsonl(qa_dir / "answers.jsonl"))
    question_sets = [{row["question_id"] for row in answers if row["config_id"] == config} for config in configs]
    if (qa_protocol["manifest_hash"] != manifest_hash or qa_protocol["configs"] != configs
            or len(answers) != 240 or len(question_sets[0]) != 120 or question_sets[0] != question_sets[1]
            or qa["human_validation"] != agreement["status"]
            or agreement["status"] not in {"validated", "exploratory"}):
        raise ValueError("Paired QA and human agreement are required for the final release")
    report = retrieval_report(dev, heldout, protocol)
    report.extend(["", "## Answer study", ""])
    for config_id, row in qa["by_config"].items():
        missed_with_evidence = next(cell["count"] for cell in row["retrieval_answer_gap"]
                                    if cell["gold_hit"] and not cell["exact_match"])
        report.append(f"- {config_id} had {row['abstentions']} abstentions across {row['questions']} questions. "
                      f"Extractive exact match was {row['extractive_exact_match']:.3f} and token F1 was "
                      f"{row['extractive_token_f1']:.3f} on {row['extractive_questions']} questions. "
                      f"{missed_with_evidence} extractive answers missed exact match despite a gold hit in context.")
    judge = agreement["judge_agreement"]
    report.extend(["", f"Judge status is **{agreement['status']}** on {judge['n']} reconciled examples. "
                   f"Judge Cohen’s κ was {judge['cohen_kappa']}.", "",
                   f"Benchmark manifest canonical JSON SHA-256 `{manifest_hash}`.", "",
                   "QASPER was created by Dasigi et al. (2021). See the dataset card for attribution and terms."])
    with atomic_file(destination / "report.md") as stream:
        stream.write("\n".join(report) + "\n")
    workspace = Path(__file__).resolve().parent.parent
    files = [(workspace / name, name) for name in ("README.md", "requirements.txt", "requirements-qa.txt",
                                                   "dashboard.py", "LICENSE")]
    files.extend((path, str(path.relative_to(workspace))) for path in
                 sorted((workspace / "retrieval_benchmark").glob("*.py")))
    files.extend((path, str(path.relative_to(workspace))) for path in sorted((workspace / "tests").glob("*.py")))
    files.extend([(Path(benchmark_root) / "manifest.json", "data/benchmark/manifest.json"),
                  (runs / "protocol.json", "runs/protocol.json"), (runs / "qa_protocol.json", "runs/qa_protocol.json"),
                  (destination / "report.md", "report.md")])
    for split in ("dev", "heldout"):
        for path in sorted((output / split).glob("*")):
            if path.is_file():
                files.append((path, f"results/{split}/{path.name}"))
        for config_id, config in CONFIGS.items():
            source = run_directory(runs, manifest_hash, split, config)
            files.extend((path, f"records/{split}/{config_id}/{path.name}")
                         for path in sorted(source.iterdir()) if path.is_file())
    archive = destination / f"retrieval-benchmark-v{VERSION}.zip"
    with atomic_file(archive, "wb") as stream:
        with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=6) as package:
            for path, name in files:
                package.write(path, name)
    with atomic_file(destination / "SHA256SUMS") as stream:
        stream.write(f"{file_hash(archive)}  {archive.name}\n")
    return archive
