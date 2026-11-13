import argparse
from pathlib import Path

from . import analysis, benchmark, qa
from .models import EMBEDDINGS, model_path
from .retrieval import CONFIGS
from .runner import retrieval_progress, run_study
from .storage import WorkBudget, fingerprint, read_json, work_lock


def main():
    parser = argparse.ArgumentParser(description="QASPER scientific-paper retrieval benchmark")
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-sources", help="Download and verify the two QASPER v0.3 archives")
    models = sub.add_parser("fetch-models", help="Cache pinned retrieval or QA models")
    models.add_argument("--qa", action="store_true")
    freeze = sub.add_parser("freeze", help="Construct the frozen benchmark")
    freeze.add_argument("--expect-hash", help="Fail if the resulting manifest hash differs")
    retrieve = sub.add_parser("retrieve", help="Run or resume retrieval experiments")
    retrieve.add_argument("split", choices=["dev", "heldout"])
    retrieve.add_argument("--config", choices=list(CONFIGS), action="append")
    retrieve.add_argument("--limit", type=int, help="Pilot the first N development questions")
    sub.add_parser("freeze-protocol", help="Lock configuration selection after the development runs")
    report = sub.add_parser("analyze", help="Export analysis for a completed split")
    report.add_argument("split", choices=["dev", "heldout"])
    answer = sub.add_parser("qa", help="Run or resume the local answer study")
    answer.add_argument("split", choices=["dev", "heldout"])
    for batched in (retrieve, answer):
        batched.add_argument("--max-jobs", type=int, help="Stop after this many new query/configuration attempts")
        batched.add_argument("--max-minutes", type=float, help="Stop between jobs when this time budget expires")
    sub.add_parser("status", help="Show checkpoint counts without running inference")
    review = sub.add_parser("export-review", help="Build a blinded human-review packet")
    review.add_argument("split", choices=["dev", "heldout"], default="heldout", nargs="?")
    reconcile = sub.add_parser("reconcile", help="Import both reviewer CSVs and calculate agreement")
    reconcile.add_argument("reviewer_1", type=Path)
    reconcile.add_argument("reviewer_2", type=Path)
    reconcile.add_argument("--resolution", type=Path)
    bundle = sub.add_parser("package-release", help="Package a checked v1.0.0 study release")
    bundle.add_argument("--destination", type=Path, default=Path("release"))
    args = parser.parse_args()

    try:
        if args.command == "fetch-sources":
            benchmark.fetch_sources(args.benchmark)
            return
        if args.command == "fetch-models":
            if args.qa:
                qa.fetch_qa_models(args.models)
            else:
                for model in EMBEDDINGS:
                    print(model, model_path(model, args.models))
            return
        if args.command == "freeze":
            manifest_hash = benchmark.freeze(args.benchmark, args.models)
            if args.expect_hash and manifest_hash != args.expect_hash:
                raise ValueError(f"Unexpected manifest hash {manifest_hash}")
            print(manifest_hash)
            return
        if args.command == "export-review":
            print(f"{qa.export_review(args.output, args.split)} blinded review items")
            return
        if args.command == "reconcile":
            print(qa.reconcile(args.output, args.reviewer_1, args.reviewer_2, args.resolution))
            return
        manifest_hash = fingerprint(read_json(args.benchmark / "manifest.json"))
        if args.command == "status":
            for split, ok, errors, total in retrieval_progress(args.benchmark, args.runs, manifest_hash):
                print(f"Retrieval {split}: {ok}/{total} complete, {errors} errors")
            for split in ("dev", "heldout"):
                ok, total = qa.qa_progress(args.runs, split)
                print(f"QA {split}: {ok}/{total} complete")
            return
        if args.command == "retrieve":
            with work_lock(args.runs, "retrieval", args.split):
                if not run_study(args.benchmark, args.runs, args.models, manifest_hash, args.split,
                                 args.config or list(CONFIGS), args.limit,
                                 WorkBudget(args.max_jobs, args.max_minutes)):
                    raise SystemExit(1)
        elif args.command == "freeze-protocol":
            protocol = analysis.freeze_protocol(args.benchmark, args.runs, manifest_hash)
            print(f"Selected {protocol['selected']}; QA compares B0 and {protocol['qa_challenger']}")
        elif args.command == "analyze":
            leaderboard = analysis.analyze(args.benchmark, args.runs, args.output, manifest_hash, args.split)
            print(f"{args.split} exported, first by Recall@5 is {leaderboard[0]['config_id']}")
        elif args.command == "qa":
            with work_lock(args.runs, "QA", args.split):
                qa.run_qa(args.benchmark, args.runs, args.models, args.output, manifest_hash, args.split,
                          budget=WorkBudget(args.max_jobs, args.max_minutes))
        elif args.command == "package-release":
            from .release import package_release
            print(package_release(args.benchmark, args.runs, args.output, args.destination, manifest_hash))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Rerun the same command to resume from the last committed checkpoint.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
