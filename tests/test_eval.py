"""Tests for the Ragas harness: benchmark loading, score mapping, exports.

The judge LLM is never called — `ragas.evaluate` is replaced by a stub that
returns the dataframe shape the real library produces, which is exactly the
contract `score_samples` depends on.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("ragas", reason="install the `eval` extra to run these tests")

from eval.eval_ragas import (
    SYNTHETIC_BENCHMARK,
    EvaluationSample,
    export_results,
    load_benchmark,
    print_report,
    score_samples,
)
from src.config import Settings


def _sample(question: str = "How long is the notice period?") -> EvaluationSample:
    return EvaluationSample(
        question=question,
        reference="Thirty (30) days written notice.",
        category="factual",
        answer="Either party may terminate with thirty (30) days notice [1].",
        contexts=["Either party may terminate the agreement with thirty (30) days notice."],
        citations=["contract.pdf p.4"],
        retrieval_ms=312.5,
        generation_ms=890.1,
        input_tokens=612,
        output_tokens=48,
    )


# --------------------------------------------------------------------------- #
# Benchmark loading
# --------------------------------------------------------------------------- #
def test_builtin_benchmark_covers_refusal_behaviour() -> None:
    items = load_benchmark(None, None)

    assert len(items) == len(SYNTHETIC_BENCHMARK)
    # A golden set without an unanswerable question cannot detect hallucination.
    assert any(item.category == "out_of_scope" for item in items)
    assert all(item.question and item.reference for item in items)


def test_limit_truncates_the_benchmark() -> None:
    assert len(load_benchmark(None, 3)) == 3


def test_custom_dataset_is_loaded_from_json(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps([{"question": "Q?", "reference": "A.", "category": "custom"}]),
        encoding="utf-8",
    )

    items = load_benchmark(path, None)

    assert items[0].question == "Q?"
    assert items[0].category == "custom"


def test_custom_dataset_must_not_be_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty JSON array"):
        load_benchmark(path, None)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_score_samples_maps_columns_and_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    import ragas

    captured: dict[str, Any] = {}

    def fake_evaluate(**kwargs: Any) -> Any:
        captured.update(kwargs)
        frame = pd.DataFrame(
            [
                {
                    "faithfulness": 1.0,
                    "answer_relevancy": 0.9,
                    "llm_context_precision_without_reference": 0.8,
                },
                {
                    "faithfulness": 0.5,
                    "answer_relevancy": float("nan"),  # a judge call that failed
                    "llm_context_precision_without_reference": 0.6,
                },
            ]
        )
        return SimpleNamespace(to_pandas=lambda: frame)

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)

    scores = score_samples([_sample(), _sample("Second question?")], "gpt-4o-mini")

    assert scores["judge_model"] == "gpt-4o-mini"
    assert scores["per_sample"][0] == {
        "faithfulness": 1.0,
        "answer_relevance": 0.9,
        "context_precision": 0.8,
    }
    # NaN becomes None instead of poisoning the mean.
    assert scores["per_sample"][1]["answer_relevance"] is None
    assert scores["aggregate"] == {
        "faithfulness": 0.75,
        "answer_relevance": 0.9,
        "context_precision": 0.7,
    }
    # The three required metrics were requested, and failures do not abort a run.
    assert len(captured["metrics"]) == 3
    assert captured["raise_exceptions"] is False
    assert len(captured["dataset"]) == 2


def test_report_renders_without_scores(capsys: pytest.CaptureFixture[str]) -> None:
    samples = [_sample()]
    scores = {
        "per_sample": [{"faithfulness": None, "answer_relevance": None, "context_precision": None}],
        "aggregate": {"faithfulness": None, "answer_relevance": None, "context_precision": None},
        "judge_model": "gpt-4o-mini",
    }

    print_report(samples, scores)

    out = capsys.readouterr().out
    assert "n/a" in out
    assert "contract.pdf" in out


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_results_are_exported_as_json_and_csv(tmp_path: Path, settings: Settings) -> None:
    samples = [_sample()]
    scores = {
        "per_sample": [{"faithfulness": 1.0, "answer_relevance": 0.93, "context_precision": 0.87}],
        "aggregate": {"faithfulness": 1.0, "answer_relevance": 0.93, "context_precision": 0.87},
        "judge_model": "gpt-4o-mini",
    }

    json_path, csv_path = export_results(samples, scores, settings, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["faithfulness"] == 1.0
    assert payload["config"]["collection"] == settings.qdrant_collection
    assert payload["config"]["reranker_model"] == settings.reranker_model
    assert payload["samples"][0]["question"] == samples[0].question

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["citations"] == "contract.pdf p.4"
    assert rows[0]["faithfulness"] == "1.0"
    assert "contexts" not in rows[0]  # long passages stay out of the flat export
