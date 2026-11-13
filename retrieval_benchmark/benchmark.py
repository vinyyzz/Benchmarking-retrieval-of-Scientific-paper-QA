from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile
from typing import Literal, Mapping, Sequence
import unicodedata
from urllib.request import urlopen

from .models import EMBEDDINGS, canonical_tokenizer, embedding_dependencies, validate_revision
from .storage import VERSION, atomic_file, file_hash, fingerprint, read_json, read_jsonl, write_json, write_jsonl

Split = Literal["dev", "heldout"]
Manifest = dict
ARCHIVES = {
    "dev": ("qasper-train-dev-v0.3.tgz", "qasper-dev-v0.3.json",
            "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"),
    "heldout": ("qasper-test-and-evaluator-v0.3.tgz", "qasper-test-v0.3.json",
                "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"),
}
SOURCE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
SPLIT_SIZE = {"dev": 60, "heldout": 120}
PUBLIC_FILES = ("papers.jsonl", "paragraphs.jsonl", "questions.jsonl", "exclusions.jsonl")
PRIVATE_FILES = ("gold.jsonl", "references.jsonl")


@dataclass(frozen=True)
class Paragraph:
    paper_id: str
    paragraph_id: str
    section_id: str
    section: str
    paragraph_order: int
    text: str
    canonical_token_count: int
    token_offsets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    paragraphs: tuple[Paragraph, ...]


@dataclass(frozen=True)
class Question:
    question_id: str
    paper_id: str
    question: str
    split: Split


@dataclass(frozen=True)
class BenchmarkView:
    manifest: Manifest
    papers: Mapping[str, Paper]
    questions: Sequence[Question]
    exclusion_counts: Mapping[str, int]


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def archive_paths(root):
    return {name: Path(root).parent / "source" / name for name, _, _ in ARCHIVES.values()}


def fetch_sources(root):
    import shutil

    for name, _, expected in ARCHIVES.values():
        path = archive_paths(root)[name]
        if path.exists() and file_hash(path) == expected:
            continue
        with urlopen(f"{SOURCE_URL}/{name}", timeout=90) as response, atomic_file(path, "wb") as stream:
            shutil.copyfileobj(response, stream)
            stream.flush()
            if file_hash(stream.name) != expected:
                raise ValueError(f"Checksum mismatch for {name}")


def _annotation(raw, by_text):
    answer = raw["answer"]
    if answer["unanswerable"]:
        return "unanswerable", None
    textual = [normalize(item) for item in answer["evidence"] if not item.startswith("FLOAT SELECTED")]
    mixed = len(textual) < len(answer["evidence"])
    if not textual:
        return ("figure_only" if mixed else "unresolved_text_evidence"), None
    if any(len(by_text.get(item, ())) != 1 for item in textual):
        return "unresolved_text_evidence", None
    if answer["extractive_spans"]:
        kind, reference = "extractive", ", ".join(answer["extractive_spans"])
    elif answer["free_form_answer"]:
        kind, reference = "abstractive", answer["free_form_answer"]
    elif isinstance(answer["yes_no"], bool):
        kind, reference = "boolean", "Yes" if answer["yes_no"] else "No"
    else:
        raise ValueError(f"Missing answer in annotation {raw['annotation_id']}")
    return None, (sorted({by_text[item][0] for item in textual}), mixed, kind, reference)


def _prepare_paper(paper_id, raw, split, tokenizer):
    paragraphs = []
    by_text = defaultdict(list)
    for section_index, section in enumerate(raw["full_text"]):
        for paragraph_index, source in enumerate(section["paragraphs"]):
            text = normalize(source)
            if not text:
                continue
            paragraph_id = f"{paper_id}:p{section_index}.{paragraph_index}"
            offsets = tuple(map(tuple, tokenizer(text, add_special_tokens=False, verbose=False,
                                                 return_offsets_mapping=True)["offset_mapping"]))
            if offsets:
                paragraph = Paragraph(paper_id, paragraph_id, f"{paper_id}:s{section_index}",
                                      normalize(section["section_name"] or ""), len(paragraphs), text,
                                      len(offsets), offsets)
                paragraphs.append(paragraph)
                by_text[text].append(paragraph_id)
    questions, gold, references, exclusions = [], [], [], []
    for qa in raw["qas"]:
        question_id = qa["question_id"]
        reasons = set()
        accepted = []
        for annotation in qa["answers"]:
            reason, result = _annotation(annotation, by_text)
            if reason:
                reasons.add(reason)
                continue
            ids, mixed, kind, reference = result
            key = {"question_id": question_id, "annotation_id": annotation["annotation_id"]}
            accepted.append({**key, "paragraph_ids": ids, "mixed_evidence": mixed})
            references.append({**key, "answer_type": kind, "answer": reference})
        if accepted:
            questions.append(Question(question_id, paper_id, qa["question"], split))
            gold.extend(accepted)
        else:
            reason = "unresolved_text_evidence"
            if reasons == {"unanswerable"}:
                reason = "unanswerable"
            elif "figure_only" in reasons and "unresolved_text_evidence" not in reasons:
                reason = "figure_only"
            exclusions.append({"question_id": question_id, "paper_id": paper_id, "reason": reason, "split": split})
    return Paper(paper_id, raw["title"], tuple(paragraphs)), questions, gold, references, exclusions


def _shuffle_key(identifier):
    return sha256(f"23:{identifier}".encode()).hexdigest(), identifier


def freeze(root, models):
    root = Path(root)
    paths = archive_paths(root)
    checksums = {}
    for name, _, expected in ARCHIVES.values():
        checksums[name] = file_hash(paths[name])
        if checksums[name] != expected:
            raise ValueError(f"Checksum mismatch for {name}")
    tokenizer = canonical_tokenizer(models)
    papers, paragraphs, questions, gold, references, exclusions = [], [], [], [], [], []
    paper_ids = {}
    for split, (name, member, _) in ARCHIVES.items():
        with tarfile.open(paths[name], "r:gz") as archive:
            with archive.extractfile(member) as stream:
                upstream = json.load(stream)
        selected = []
        for paper_id in sorted(upstream, key=_shuffle_key):
            paper, qa, evidence, answers, excluded = _prepare_paper(paper_id, upstream[paper_id], split, tokenizer)
            if qa:
                selected.append(paper_id)
                papers.append({"paper_id": paper_id, "title": paper.title, "split": split})
                paragraphs.extend(map(asdict, paper.paragraphs))
                questions.extend(map(asdict, qa))
                gold.extend(evidence)
                references.extend(answers)
                exclusions.extend(excluded)
            if len(selected) == SPLIT_SIZE[split]:
                break
        if len(selected) != SPLIT_SIZE[split]:
            raise ValueError(f"Insufficient eligible papers in {split}")
        paper_ids[split] = selected
    if set(paper_ids["dev"]) & set(paper_ids["heldout"]):
        raise ValueError("Paper leakage across splits")
    for name, rows in zip(PUBLIC_FILES + PRIVATE_FILES, (papers, paragraphs, questions, exclusions, gold, references),
                          strict=True):
        write_jsonl(root / name, rows)
    revisions = {model: revision for model, (_, revision) in EMBEDDINGS.items()}
    manifest = {
        "api_version": VERSION, "schema_version": VERSION, "dataset": "QASPER v0.3",
        "paper_ids_by_split": paper_ids, "source_checksums": checksums,
        "preprocessing_version": VERSION, "sentence_rule_version": VERSION, "evaluator_version": VERSION,
        "canonical_tokenizer_revision": revisions["E1"], "model_revisions": revisions,
        "tokenizer_revisions": revisions, "dependencies": embedding_dependencies(),
        "artifact_checksums": {name: file_hash(root / name) for name in PUBLIC_FILES + PRIVATE_FILES},
        "question_counts": dict(Counter(row["split"] for row in questions)),
        "exclusion_counts": {split: dict(Counter(row["reason"] for row in exclusions if row["split"] == split))
                             for split in SPLIT_SIZE},
    }
    write_json(root / "manifest.json", manifest)
    return fingerprint(manifest)


def verified_manifest(root, manifest_hash, files):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if fingerprint(manifest) != manifest_hash:
        raise ValueError("Manifest hash mismatch")
    for key in ("schema_version", "api_version", "preprocessing_version", "sentence_rule_version",
                "evaluator_version"):
        if manifest.get(key) != VERSION:
            raise ValueError(f"Incompatible {key}")
    for key in ("model_revisions", "tokenizer_revisions"):
        if set(manifest[key]) != set(EMBEDDINGS) or not all(map(validate_revision, manifest[key].values())):
            raise ValueError(f"Missing immutable revisions in {key}")
    if manifest["canonical_tokenizer_revision"] != manifest["tokenizer_revisions"]["E1"]:
        raise ValueError("Canonical tokenizer revision mismatch")
    paths = archive_paths(root)
    if set(paths) != set(manifest["source_checksums"]):
        raise ValueError("Unexpected source archive list")
    for name, checksum in manifest["source_checksums"].items():
        if file_hash(paths[name]) != checksum:
            raise ValueError(f"Source checksum mismatch for {name}")
    for name in files:
        if file_hash(root / name) != manifest["artifact_checksums"][name]:
            raise ValueError(f"Artifact checksum mismatch for {name}")
    return manifest


def load_benchmark(root: Path, split: Split, *, manifest_hash: str) -> BenchmarkView:
    root = Path(root)
    manifest = verified_manifest(root, manifest_hash, PUBLIC_FILES)
    partitions = manifest["paper_ids_by_split"]
    if split not in SPLIT_SIZE or set(partitions) != set(SPLIT_SIZE):
        raise ValueError("Invalid paper split")
    for key, expected in SPLIT_SIZE.items():
        if len(partitions[key]) != expected or len(set(partitions[key])) != expected:
            raise ValueError(f"Invalid paper count in {key}")
    if set(partitions["dev"]) & set(partitions["heldout"]):
        raise ValueError("Paper leakage across splits")
    paper_split = {paper_id: key for key, ids in partitions.items() for paper_id in ids}
    texts = defaultdict(list)
    titles = {}
    seen = set()
    for row in read_jsonl(root / "papers.jsonl"):
        paper_id = row["paper_id"]
        if paper_id in titles or paper_split.get(paper_id) != row["split"]:
            raise ValueError(f"Invalid paper {paper_id}")
        titles[paper_id] = row["title"]
    if set(titles) != set(paper_split):
        raise ValueError("Paper records differ from manifest")
    for row in read_jsonl(root / "paragraphs.jsonl"):
        row.pop("schema_version")
        row["token_offsets"] = tuple(map(tuple, row["token_offsets"]))
        paragraph = Paragraph(**row)
        valid_offsets = all(0 <= start < end <= len(paragraph.text) for start, end in paragraph.token_offsets)
        if (paragraph.paper_id not in titles or paragraph.paragraph_id in seen or not paragraph.text
                or paragraph.paragraph_order != len(texts[paragraph.paper_id])
                or paragraph.canonical_token_count != len(paragraph.token_offsets)
                or not paragraph.canonical_token_count or not valid_offsets):
            raise ValueError(f"Invalid paragraph {paragraph.paragraph_id}")
        seen.add(paragraph.paragraph_id)
        texts[paragraph.paper_id].append(paragraph)
    questions = []
    seen.clear()
    for row in read_jsonl(root / "questions.jsonl"):
        row.pop("schema_version")
        question = Question(**row)
        if (question.question_id in seen or paper_split.get(question.paper_id) != question.split
                or not question.question.strip() or not texts[question.paper_id]):
            raise ValueError(f"Invalid question {question.question_id}")
        seen.add(question.question_id)
        if question.split == split:
            questions.append(question)
    if len(questions) != manifest["question_counts"][split]:
        raise ValueError("Question count differs from manifest")
    papers = {key: Paper(key, titles[key], tuple(texts[key])) for key in partitions[split]}
    return BenchmarkView(manifest, papers, tuple(questions), manifest["exclusion_counts"][split])
