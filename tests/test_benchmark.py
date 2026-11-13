from dataclasses import asdict
import csv
from math import log1p
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from retrieval_benchmark.analysis import _selection_key, paired_bootstrap
from retrieval_benchmark.benchmark import Paper, Paragraph, Question, _annotation, normalize
from retrieval_benchmark.evaluator import Evaluator, RankingError
from retrieval_benchmark.qa import (_assembled_context, agreement, export_review, extractive_scores,
                                   reconcile, stratified, summarize_qa)
from retrieval_benchmark.retrieval import (BM25Retriever, Chunk, ChunkBuilder, ChunkPolicy,
                                            DenseRetriever, HybridRetriever, PaperIndex, RankedChunk,
                                            Span, sparse_data)
from retrieval_benchmark.runner import load_query, query_path
from retrieval_benchmark.storage import VERSION, WorkBudget, read_json, read_jsonl, work_lock, write_json, write_jsonl


def paragraph(number, size=100, section="s1"):
    words = "x" * size
    return Paragraph("paper", f"P{number}", section, section, number, words, size,
                     tuple((i, i + 1) for i in range(size)))


def chunk(identifier, source, start=0, stop=None):
    stop = source.canonical_token_count if stop is None else stop
    return Chunk(identifier, source.paper_id, source.text[start:stop],
                 (Span(source.paper_id, source.paragraph_id, start, stop, start, stop),), stop - start)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.paragraphs = {i: paragraph(i) for i in (4, 7, 9, 11)}
        data = {p.paragraph_id: asdict(p) for p in self.paragraphs.values()}
        self.evaluator = Evaluator({"Q": "paper", "Q2": "paper"}, data,
                                   {"Q": [frozenset(("P7", "P11"))],
                                    "Q2": [frozenset(("P7", "P11")), frozenset(("P7",))]},
                                   {"Q": [("extractive", "answer")], "Q2": [("extractive", "answer")]})
        self.chunks = [chunk(str(i), self.paragraphs[p], 0, stop)
                       for i, (p, stop) in enumerate([(4, 100), (7, 70), (9, 100), (7, 70), (11, 100)], 1)]
        self.evaluator.register_chunks("chunks", self.chunks)

    def score(self, ids, question="Q", artifact="chunks"):
        ranks = tuple(RankedChunk(i, chunk_id, 1.0 / i) for i, chunk_id in enumerate(ids, 1))
        return self.evaluator.score_query("run", question, ranks, chunk_artifact_id=artifact)

    def test_worked_example(self):
        rows = self.score(["1", "2", "3", "4", "5"])
        self.assertEqual([row.recall for row in rows], [0, 0.5, 1, 1])
        self.assertEqual([row.RR for row in rows], [0.5] * 4)

    def test_annotation_average_and_empty_rank(self):
        self.assertEqual(self.score(["2"], question="Q2")[0].recall, 0.75)
        self.assertEqual(self.score([])[0].recall, 0)
        self.assertEqual(self.score([])[0].RR, 0)

    def test_duplicate_stale_and_nonfinite_ranks(self):
        for ids, artifact in [(["2", "2"], "chunks"), (["bad"], "chunks"), (["2"], "stale")]:
            with self.subTest(ids=ids, artifact=artifact), self.assertRaises(RankingError):
                self.score(ids, artifact=artifact)
        with self.assertRaises(RankingError):
            self.evaluator.score_query("run", "Q", [RankedChunk(1, "2", float("nan"))],
                                       chunk_artifact_id="chunks")

    def test_fragmentation_is_diagnostic_only(self):
        chunks = [chunk("a", self.paragraphs[7], 0, 30), chunk("b", self.paragraphs[7], 30, 60)]
        self.evaluator.register_chunks("fragments", chunks)
        rows = self.score(["a", "b"], artifact="fragments")
        self.assertEqual(rows[1].recall, 0)
        ranks = [RankedChunk(1, "a", 1), RankedChunk(2, "b", 0.5)]
        diagnostics = list(self.evaluator.diagnostics("Q2", ranks, chunk_artifact_id="fragments"))
        self.assertAlmostEqual(diagnostics[1]["unique_gold_token_coverage"], 0.45)
        self.assertEqual(diagnostics[1]["recall_threshold_25"], 0.75)

    def test_provenance_must_render_exactly(self):
        damaged = Chunk("bad", "paper", "changed", self.chunks[0].spans, self.chunks[0].token_count)
        with self.assertRaises(RankingError):
            self.evaluator.register_chunks("other", [damaged])

    def test_mixed_evidence_and_normalization(self):
        answer = {"unanswerable": False, "evidence": ["  Ｈello ", "FLOAT SELECTED figure"],
                  "extractive_spans": [], "free_form_answer": "", "yes_no": False}
        source = {"annotation_id": "a", "answer": answer}
        self.assertEqual(normalize("  Ｈello "), "Hello")
        self.assertEqual(_annotation(source, {"Hello": ["P1"]}), (None, (["P1"], True, "boolean", "No")))
        self.assertEqual(_annotation(source, {"Hello": ["P1", "P2"]})[0], "unresolved_text_evidence")


class RetrievalTests(unittest.TestCase):
    def test_chunk_strategies_cover_every_token_once(self):
        class OrthogonalEncoder:
            def encode(self, texts):
                return np.eye(len(texts))

        paper = Paper("paper", "Title", (paragraph(1, 245), paragraph(2, 160, "s2")))
        for strategy in ("fixed", "semantic", "section_aware"):
            with self.subTest(strategy=strategy):
                chunks = ChunkBuilder(OrthogonalEncoder()).build(paper, ChunkPolicy(strategy))
                spans = [span for item in chunks for span in item.spans]
                self.assertEqual(sum(item.token_count for item in chunks), 405)
                self.assertTrue(all(item.token_count <= 240 for item in chunks))
                for p in paper.paragraphs:
                    tokens = [i for span in spans if span.paragraph_id == p.paragraph_id
                              for i in range(span.token_start, span.token_end)]
                    self.assertEqual(tokens, list(range(p.canonical_token_count)))
                if strategy == "fixed":
                    self.assertEqual([c.token_count for c in chunks], [200, 200, 5])
                else:
                    self.assertTrue(all(len({span.paragraph_id for span in c.spans}) == 1 for c in chunks))

    def test_bm25_ties_and_scope(self):
        chunks = tuple(Chunk(key, "paper", text, (), 1) for key, text in [("b", "TEST"), ("a", "test"), ("c", "other")])
        index = PaperIndex("paper", "manifest", "chunks", "index", chunks, sparse_data(chunks))
        question = Question("Q", "paper", "test TEST", "dev")
        ranking = BM25Retriever().search(question, index=index)
        self.assertEqual([row.chunk_id for row in ranking], ["a", "b", "c"])
        self.assertAlmostEqual(ranking[0].score, log1p(1.5 / 2.5))
        with self.assertRaises(ValueError):
            BM25Retriever().search(Question("Q", "other", "test", "dev"), index=index)

    def test_dense_and_rrf_candidate_cutoffs(self):
        class FixedEncoder:
            def encode(self, texts):
                return np.array([[1.0, 0.0]])

        chunks = tuple(Chunk(f"{i:02}", "paper", "match", (), 1) for i in range(60))
        index = PaperIndex("paper", "manifest", "chunks", "index", chunks, sparse_data(chunks),
                           np.array([[i / 60, 0.0] for i in range(60)]))
        question = Question("Q", "paper", "match", "dev")
        self.assertEqual(DenseRetriever(FixedEncoder()).search(question, index=index)[0].chunk_id, "59")
        hybrid = HybridRetriever(FixedEncoder()).search(question, index=index, limit=50)
        scores = {row.chunk_id: row.score for row in hybrid}
        self.assertAlmostEqual(scores["10"], 1 / 71 + 1 / 110)
        self.assertAlmostEqual(scores["00"], 1 / 61)
        self.assertAlmostEqual(scores["59"], 1 / 61)
        self.assertNotIn("09", scores)
        self.assertEqual(len(scores), 50)

    def test_incomplete_checkpoint_is_not_reused(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder) / "run"
            path = query_path(directory, "q")
            write_json(path, {"run_id": "run", "question_id": "q", "schema_version": VERSION,
                              "status": "ok", "metrics": []})
            self.assertIsNone(load_query(directory, "q"))
            write_json(path, {"run_id": "run", "question_id": "q", "schema_version": VERSION,
                              "status": "retrieval_error", "attempt": 1})
            self.assertEqual(load_query(directory, "q")["attempt"], 1)


class AnalysisAndQATests(unittest.TestCase):
    def test_work_budget_stops_at_first_reached_limit(self):
        with patch("retrieval_benchmark.storage.monotonic", side_effect=[100, 110, 130]):
            budget = WorkBudget(max_jobs=3, max_minutes=0.5)
            self.assertTrue(budget.take())
            self.assertFalse(budget.take())
        budget = WorkBudget(max_jobs=1)
        self.assertTrue(budget.take())
        self.assertFalse(budget.take())
        for limit in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                WorkBudget(max_minutes=limit)

    def test_work_lock_blocks_only_the_same_partition_and_releases(self):
        with TemporaryDirectory() as folder:
            with work_lock(folder, "QA", "dev"):
                with self.assertRaisesRegex(ValueError, "already active"):
                    with work_lock(folder, "QA", "dev"):
                        self.fail("A duplicate lock was acquired")
                with work_lock(folder, "QA", "heldout"):
                    pass
            with work_lock(folder, "QA", "dev"):
                pass

    def test_paired_bootstrap_uses_paper_clusters(self):
        lower, upper = paired_bootstrap([1.0, 1.0, -1.0], ["A", "A", "B"])
        self.assertEqual((lower, upper), (-1, 1))
        self.assertEqual(paired_bootstrap([0.0, 0.0], ["A", "B"]), (0, 0))

    def test_selection_prefers_baseline_only_after_metric_ties(self):
        rows = [{"config_id": name, "recall@5": recall, "mrr@10": rr}
                for name, recall, rr in [("B0", .8, .7), ("A", .8, .7), ("C", .8, .75)]]
        self.assertEqual([row["config_id"] for row in sorted(rows, key=_selection_key)], ["C", "B0", "A"])

    def test_stratified_sample_is_order_independent(self):
        items = [(str(i), "small" if i < 10 else "big") for i in range(50)]
        select = lambda values: stratified(values, 20, lambda row: row[1], lambda row: row[0], "test")
        self.assertEqual(select(items), select(list(reversed(items))))
        self.assertEqual(sum(row[1] == "small" for row in select(items)), 4)

    def test_extractive_match_and_abstention(self):
        self.assertEqual(extractive_scores("The value, 23.", ["value 23"]), (1, 1))
        self.assertEqual(extractive_scores("23", ["value 23"]), (0, 2 / 3))
        self.assertEqual(extractive_scores("", ["value 23"]), (0, 0))
        self.assertEqual(extractive_scores("23", []), (None, None))

    def test_agreement_and_undefined_kappa(self):
        labels = ["supported", "supported", "unsupported", "unsupported"]
        result = agreement(labels, ["supported", "unsupported", "unsupported", "unsupported"])
        self.assertEqual(result["percent_agreement"], 75)
        self.assertEqual(result["cohen_kappa"], 0.5)
        self.assertIsNone(agreement(["supported"], ["supported"])["cohen_kappa"])

    def test_context_drops_whole_chunks_in_rank_order(self):
        class CharTokenizer:
            def token_count(self, text):
                return len(text)

        chunks = [Chunk(str(i), "paper", "x" * size, (), 1) for i, size in enumerate((1000, 1500, 100))]
        context = _assembled_context(chunks, CharTokenizer(), CharTokenizer())
        self.assertEqual([row["label"] for row in context], ["C1"])

    def test_review_packet_is_blinded_and_reconciliation_uses_both_reviewers(self):
        rows = [
            {"config_id": "B0", "question_id": key, "question": "<test>", "answer": "test",
             "answer_type": "extractive", "cited_passages": [], "abstained": False,
             "exact_match": 1.0, "token_f1": 1.0, "gold_hit_in_context": True, "judge_verdict": verdict}
            for key, verdict in (("q1", "supported"), ("q2", "unsupported"))
        ]
        with TemporaryDirectory() as folder:
            directory = Path(folder) / "heldout"
            write_jsonl(directory / "answers.jsonl", rows)
            write_json(directory / "qa_summary.json", summarize_qa(rows, ["B0"]))
            self.assertEqual(export_review(folder), 2)
            review = directory / "review"
            for item in read_jsonl(review / "packet.jsonl"):
                self.assertEqual(set(item), {"schema_version", "review_id", "question", "answer", "cited_passages"})
            self.assertIn("&lt;test&gt;", (review / "packet.html").read_text())
            mapping = read_json(review / "private_map.json")
            ids = list(mapping)
            first = review / "reviewer_1.csv"
            second = review / "reviewer_2.csv"

            def label_file(path, labels):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("review_id", "label"))
                    writer.writerows(labels)

            label_file(first, [(key, mapping[key]["judge_verdict"]) for key in ids])
            label_file(second, [(key, "unsupported") for key in ids])
            pending = reconcile(folder, first, second)
            self.assertEqual(pending["status"], "reconciliation_required")
            disputed = next(key for key in ids if mapping[key]["judge_verdict"] == "supported")
            resolution = Path(pending["template"])
            label_file(resolution, [(disputed, "supported")])
            result = reconcile(folder, first, second, resolution)
            self.assertEqual(result["status"], "exploratory")
            self.assertEqual(result["reviewer_agreement"]["percent_agreement"], 50)
            self.assertEqual(result["judge_agreement"]["cohen_kappa"], 1)


if __name__ == "__main__":
    unittest.main()
