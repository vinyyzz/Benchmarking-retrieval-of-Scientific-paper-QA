# Benchmarking Retrieval for Scientific-Paper QA

This benchmark measures how chunking, retrieval, embeddings, and context size affect scientific-paper question answering. The primary study searches within each question’s associated paper. It uses 60 QASPER development papers for configuration selection and 120 held-out papers for evaluation.

## Setup and first baseline

Use Python 3.12 from the project directory. In Colab or Kaggle, use the same Python version and run the commands in a shell cell. Keep the working directories on persistent storage to resume interrupted sessions. Model weights are downloaded once. The study runs on CPU without training or an API key.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m retrieval_benchmark fetch-sources
python -m retrieval_benchmark fetch-models
python -m retrieval_benchmark freeze
python -m retrieval_benchmark retrieve dev --config B0 --limit 1
python -m unittest discover -s tests -v
```

For a byte-for-byte rebuild, pass `freeze --expect-hash 976ac6b3df87155bb2dcb446dde5adfc7afd9ef870d574bb24c592babc316a13`. The benchmark root and its sibling `source` directory must stay together. Use the global `--benchmark`, `--models`, `--runs`, and `--output` options before the command to change their locations.

## Continuous integration

The Benchmark CI workflow runs the fixture-based unit tests on Python 3.12 for pushes and pull requests. It installs NumPy at the version pinned in `requirements.txt`. The tests use temporary files and fixtures, so the workflow needs no QASPER archives, model weights, or reviewer data.

## Run the retrieval study

```bash
python -m retrieval_benchmark retrieve dev
python -m retrieval_benchmark freeze-protocol
python -m retrieval_benchmark analyze dev
python -m retrieval_benchmark retrieve heldout
python -m retrieval_benchmark analyze heldout
```

Run the same retrieval command again to resume a partial run. Each query's ranking, scores, and status are committed together. An eligible question with a valid empty ranking scores zero. Processing failures retain null scores and block a complete analysis. Set `HF_HUB_OFFLINE=1` after caching the embedding models to run without network access.

## Open the dashboard on localhost

From the project root, start Streamlit using the existing results.

```bash
.venv/bin/python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open [the local dashboard](http://127.0.0.1:8501). Keep the terminal running while you use it. Press Ctrl+C to stop it. The dashboard reads the saved retrieval and QA exports, so you can reopen it without rerunning inference. Set `RETRIEVAL_RESULTS` to point Streamlit at a different export directory.

The `k` settings reuse the same ranking as `B0`. MRR@10 always uses the full ranking, so its value stays the same across those settings. The held-out tables include paired paper-cluster bootstrap intervals, cost measures, and evidence-coverage diagnostics. The dashboard loads the exported tables and traces without model inference.

## Work in short sessions

Read [PROGRESS.md](PROGRESS.md) for the current task and [CLAUDE.md](CLAUDE.md) for the project architecture. Check checkpoints before starting another runner on the same partition.

```bash
python -m retrieval_benchmark status
python -m retrieval_benchmark retrieve dev --max-jobs 100 --max-minutes 10
python -m retrieval_benchmark qa heldout --max-jobs 24 --max-minutes 15
```

The first limit reached ends the batch successfully at a checkpoint. The time limit is checked between jobs, so the current job may extend it. Repeat the same command to skip completed jobs. `Ctrl-C` also preserves prior checkpoints. Run `analyze` or `export-review` after the corresponding partition reaches its full count.

## Run answer generation and human review

Install the optional llama.cpp bindings. The pinned package can build from source and needs a working C/C++ toolchain. On macOS, install Xcode Command Line Tools if a compiler is missing. The CPU build below uses the same execution policy on a laptop or a Python 3.12 notebook.

```bash
CMAKE_ARGS=-DGGML_METAL=OFF python -m pip install -r requirements-qa.txt
python -m retrieval_benchmark fetch-models --qa
python -m retrieval_benchmark qa dev
python -m retrieval_benchmark export-review dev
python -m retrieval_benchmark qa heldout
python -m retrieval_benchmark export-review
```

The QA download command resumes partial files. If the download is interrupted repeatedly, place the official
`qwen2.5-1.5b-instruct-q4_k_m.gguf` and `granite-3.3-2b-instruct-Q4_K_M.gguf` files in `models/qa/`.
The QA command checks both against the pinned SHA-256 values before inference.
[Qwen model files](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9),
[Granite model files](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/tree/7cdf86ccd1f1bb3491c9b7017b033f2e51367397)

The QA pilot checks local inference and writes a frozen model and prompt protocol. The held-out command evaluates the same 120 questions with `B0` and the best nonbaseline development setting. A valid empty answer is an abstention. Extractive EM and F1 use the best matching accepted extractive reference. QA and retrieval results use different denominators, shown in the exports.

Have both reviewers practice with the development packet. Give each reviewer the held-out `results/heldout/review/packet.html` and their own CSV. Each person fills the label column independently with `supported` or `unsupported`. Keep `private_map.json` separate until review ends. The judge and reviewers evaluate support in the cited passages, independent of answer completeness.

```bash
python -m retrieval_benchmark reconcile \
  results/heldout/review/reviewer_1.csv results/heldout/review/reviewer_2.csv
```

If the command creates `reconciliation.csv`, have the reviewers agree on the disputed items and fill its label column. Repeat the same command with `--resolution results/heldout/review/reconciliation.csv`. The output includes the independent reviewer agreement and the judge’s agreement with reconciled human labels.

## Package the public release

```bash
python -m retrieval_benchmark package-release
```

The command requires both complete retrieval partitions, the paired held-out QA study, and completed human reconciliation. Upload the generated archive, checksum, and report to the project’s public GitHub repository under tag `v1.0.0`. The archive includes the frozen manifest and configuration suite, versioned result files, and the code to regenerate benchmark records from the verified upstream archives.

## Dataset and scope

QASPER was created by Dasigi et al. The dataset card lists CC BY 4.0. The project normalizes paragraph text, samples papers, and evaluates resolvable text evidence. Mixed annotations retain only their text evidence for retrieval scoring. The release includes exclusion counts, and the benchmark is scoped to English NLP papers and source-paper search. [QASPER dataset card](https://huggingface.co/datasets/allenai/qasper), [QASPER paper](https://aclanthology.org/2021.naacl-main.365/)
