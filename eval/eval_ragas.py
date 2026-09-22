"""Ragas evaluation harness: does the pipeline actually answer from the documents?

The benchmark runs the *production* retrieval and generation code against a
synthetic golden set, then scores every sample with LLM-as-a-judge metrics:

    faithfulness      — is every claim in the answer supported by the retrieved
                        context? (hallucination detector)
    answer_relevance   — does the answer actually address the question?
    context_precision  — are the retrieved passages relevant, and ranked well?
                        (this is the metric that proves the reranker earns its
                        latency — compare with RERANK_TOP_N=k vs. dense-only)

Answers come from the configured `LLM_PROVIDER` (OpenAI or a local Ollama),
while the judge is always OpenAI: grading is only comparable across runs with a
strong, stable evaluator, and answer relevance needs OpenAI embeddings.

Usage
-----
    uv run python -m eval.eval_ragas                 # full run (needs OpenAI + Qdrant)
    uv run python -m eval.eval_ragas --dry-run       # validate wiring, no API calls
    uv run python -m eval.eval_ragas --limit 3 --k 5
    uv run python -m eval.eval_ragas --dataset eval/my_golden_set.json

Results are printed and written to `eval/results/<timestamp>.{json,csv}`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table
from src.config import PROJECT_ROOT, Settings, get_settings
from src.main import NO_CONTEXT_ANSWER, build_llm_client, build_messages
from src.retriever import (
    TwoStageRetriever,
    build_qdrant_client,
    load_embedding_model,
    load_reranker,
)

logger = logging.getLogger("eval.ragas")
console = Console()

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
MAX_CONCURRENCY = 4

# Metric column name in Ragas -> label used in the report.
METRIC_LABELS: dict[str, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevance",
    "llm_context_precision_without_reference": "context_precision",
}


# --------------------------------------------------------------------------- #
# Synthetic benchmark
#
# Questions are written against `documents/sample_policy_handbook.pdf`, which
# ships with the repository, so `uv run python -m eval.eval_ragas` works out of
# the box after ingestion. Replace with a domain golden set via --dataset.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One golden-set question with its reference answer."""

    question: str
    reference: str
    category: str = "factual"


SYNTHETIC_BENCHMARK: tuple[BenchmarkItem, ...] = (
    BenchmarkItem(
        question="How many days of notice are required to terminate the service agreement?",
        reference="Either party may terminate the agreement with thirty (30) days written notice.",
    ),
    BenchmarkItem(
        question="Within how many days must invoices be paid?",
        reference="Invoices are payable within fourteen (14) days of the invoice date.",
    ),
    BenchmarkItem(
        question="What is the annual leave entitlement for full-time employees?",
        reference="Full-time employees receive twenty-six (26) working days of paid annual leave "
        "per calendar year.",
    ),
    BenchmarkItem(
        question="How quickly must a suspected security incident be reported?",
        reference="Suspected security incidents must be reported to the Security Operations "
        "Centre within one hour of detection.",
    ),
    BenchmarkItem(
        question="What is the uptime commitment in the service level agreement?",
        reference="The service level agreement commits to 99.5% monthly uptime, excluding "
        "scheduled maintenance windows.",
    ),
    BenchmarkItem(
        question="How long is personal data retained after an account is closed?",
        reference="Personal data is deleted or anonymised within ninety (90) days of account "
        "closure, unless a legal retention obligation applies.",
    ),
    BenchmarkItem(
        question="What is the maximum liability cap under the agreement?",
        reference="Aggregate liability is capped at the total fees paid in the twelve (12) "
        "months preceding the claim.",
    ),
    BenchmarkItem(
        question="Who must approve an exception to the remote working policy?",
        reference="Exceptions to the remote working policy require written approval from the "
        "department head and the People Operations lead.",
    ),
    BenchmarkItem(
        question="What is the price of a corporate helicopter charter?",
        reference=NO_CONTEXT_ANSWER,
        category="out_of_scope",  # must trigger a refusal, not a hallucination
    ),
)


@dataclass(slots=True)
class EvaluationSample:
    """A generated answer with the context it was grounded in."""

    question: str
    reference: str
    category: str
    answer: str
    contexts: list[str]
    citations: list[str]
    retrieval_ms: float
    generation_ms: float
    input_tokens: int
    output_tokens: int


# --------------------------------------------------------------------------- #
# Benchmark loading
# --------------------------------------------------------------------------- #
def load_benchmark(path: Path | None, limit: int | None) -> list[BenchmarkItem]:
    """Load the golden set from disk, or fall back to the built-in synthetic one."""
    if path is None:
        items = list(SYNTHETIC_BENCHMARK)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            msg = f"{path} must contain a non-empty JSON array of objects"
            raise ValueError(msg)
        items = [
            BenchmarkItem(
                question=str(entry["question"]),
                reference=str(entry["reference"]),
                category=str(entry.get("category", "factual")),
            )
            for entry in raw
        ]
    return items[:limit] if limit else items


# --------------------------------------------------------------------------- #
# Answer generation (production code path)
# --------------------------------------------------------------------------- #
async def generate_sample(
    item: BenchmarkItem,
    retriever: TwoStageRetriever,
    client: AsyncOpenAI,
    settings: Settings,
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> EvaluationSample:
    """Retrieve, generate and record one benchmark sample."""
    async with semaphore:
        retrieval = await retriever.retrieve(item.question, top_n=top_k)
        contexts = [chunk.text for chunk in retrieval.chunks]
        citations = [chunk.citation() for chunk in retrieval.chunks]

        if not retrieval.chunks:
            return EvaluationSample(
                question=item.question,
                reference=item.reference,
                category=item.category,
                answer=NO_CONTEXT_ANSWER,
                contexts=[],
                citations=[],
                retrieval_ms=retrieval.total_ms,
                generation_ms=0.0,
                input_tokens=0,
                output_tokens=0,
            )

        started = time.perf_counter()
        completion = await client.chat.completions.create(
            model=settings.llm_model,
            messages=build_messages(item.question, retrieval.chunks),
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
        )
        generation_ms = (time.perf_counter() - started) * 1000
        usage = completion.usage
        return EvaluationSample(
            question=item.question,
            reference=item.reference,
            category=item.category,
            answer=completion.choices[0].message.content or "",
            contexts=contexts,
            citations=citations,
            retrieval_ms=retrieval.total_ms,
            generation_ms=generation_ms,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


async def collect_samples(
    items: Sequence[BenchmarkItem], settings: Settings, top_k: int
) -> list[EvaluationSample]:
    """Run the whole benchmark through the real retrieval + generation stack."""
    encoder, reranker = await asyncio.gather(
        asyncio.to_thread(load_embedding_model, settings),
        asyncio.to_thread(load_reranker, settings),
    )
    retriever = TwoStageRetriever(
        client=build_qdrant_client(settings),
        encoder=encoder,
        reranker=reranker,
        settings=settings,
    )
    client = build_llm_client(settings)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    try:
        with console.status(f"Generating {len(items)} answers…", spinner="dots"):
            return await asyncio.gather(
                *(
                    generate_sample(item, retriever, client, settings, top_k, semaphore)
                    for item in items
                )
            )
    finally:
        await client.close()
        await retriever.close()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_samples(samples: Sequence[EvaluationSample], judge_model: str) -> dict[str, Any]:
    """Score generated answers with Ragas; returns per-sample and aggregate scores."""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    hf_dataset = Dataset.from_list(
        [
            {
                "user_input": sample.question,
                "retrieved_contexts": sample.contexts,
                "response": sample.answer,
                "reference": sample.reference,
            }
            for sample in samples
        ]
    )
    dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=judge),
    ]

    # `evaluate` is typed as returning a union with `Executor`; with
    # `return_executor` left at its default we always get an EvaluationResult.
    result: Any = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=MAX_CONCURRENCY, timeout=180),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    per_sample: list[dict[str, float | None]] = []
    for _, row in frame.iterrows():
        scores: dict[str, float | None] = {}
        for column, label in METRIC_LABELS.items():
            value = row.get(column)
            scores[label] = None if value is None or value != value else float(value)
        per_sample.append(scores)

    aggregate: dict[str, float | None] = {}
    for label in METRIC_LABELS.values():
        values = [row[label] for row in per_sample if row[label] is not None]
        aggregate[label] = round(sum(values) / len(values), 4) if values else None  # type: ignore[arg-type]
    return {"per_sample": per_sample, "aggregate": aggregate, "judge_model": judge_model}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: float | None) -> str:
    if value is None:
        return "[dim]n/a[/dim]"
    colour = "green" if value >= 0.8 else "yellow" if value >= 0.6 else "red"
    return f"[{colour}]{value:.3f}[/{colour}]"


def print_report(samples: Sequence[EvaluationSample], scores: dict[str, Any]) -> None:
    """Render per-sample and aggregate results to the console."""
    table = Table(title="Ragas benchmark — per sample", header_style="bold", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Question", max_width=44, overflow="ellipsis")
    table.add_column("Faith.", justify="right")
    table.add_column("Ans.rel.", justify="right")
    table.add_column("Ctx.prec.", justify="right")
    table.add_column("Sources", max_width=22, overflow="ellipsis")
    table.add_column("ms", justify="right")

    for index, (sample, row) in enumerate(zip(samples, scores["per_sample"], strict=True), 1):
        table.add_row(
            str(index),
            sample.question,
            _fmt(row["faithfulness"]),
            _fmt(row["answer_relevance"]),
            _fmt(row["context_precision"]),
            ", ".join(sample.citations) or "—",
            f"{sample.retrieval_ms + sample.generation_ms:.0f}",
        )
    console.print(table)

    summary = Table(title="Aggregate", header_style="bold")
    summary.add_column("Metric")
    summary.add_column("Score", justify="right")
    for label, value in scores["aggregate"].items():
        summary.add_row(label, _fmt(value))
    latency = [s.retrieval_ms + s.generation_ms for s in samples]
    summary.add_section()
    summary.add_row("samples", str(len(samples)))
    summary.add_row("mean latency (ms)", f"{sum(latency) / max(len(latency), 1):.0f}")
    summary.add_row(
        "mean retrieval (ms)",
        f"{sum(s.retrieval_ms for s in samples) / max(len(samples), 1):.0f}",
    )
    summary.add_row("judge model", scores["judge_model"])
    console.print(summary)


def export_results(
    samples: Sequence[EvaluationSample],
    scores: dict[str, Any],
    settings: Settings,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist the benchmark run as JSON (full) and CSV (per-sample)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"ragas_{stamp}.json"
    csv_path = output_dir / f"ragas_{stamp}.csv"

    rows = [
        {**asdict(sample), **score}
        for sample, score in zip(samples, scores["per_sample"], strict=True)
    ]
    payload = {
        "run_at": stamp,
        "config": {
            "generation_provider": settings.llm_provider,
            "generation_model": settings.llm_model,
            "judge_model": scores["judge_model"],
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "retrieval_candidates": settings.retrieval_candidates,
            "rerank_top_n": settings.rerank_top_n,
            "collection": settings.qdrant_collection,
        },
        "aggregate": scores["aggregate"],
        "samples": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "question",
        "category",
        "answer",
        "reference",
        "citations",
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "retrieval_ms",
        "generation_ms",
        "input_tokens",
        "output_tokens",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "citations": "; ".join(row["citations"])})

    console.print(f"[green]✓[/green] Results written to [cyan]{json_path}[/cyan] and CSV")
    return json_path, csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval-ragas",
        description="Evaluate the RAG pipeline with Ragas (faithfulness, answer relevance, "
        "context precision).",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Custom golden set (JSON).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items.")
    parser.add_argument(
        "--k", type=int, default=None, help="Passages per question (default: config)."
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenAI model used as the Ragas judge (default: OPENAI_MODEL).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Where to write results."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and benchmark integrity without calling any API.",
    )
    return parser.parse_args(argv)


def _dry_run(items: Sequence[BenchmarkItem], settings: Settings, top_k: int) -> int:
    table = Table(title="Benchmark (dry run)", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Category")
    table.add_column("Question", max_width=70, overflow="fold")
    for index, item in enumerate(items, 1):
        table.add_row(str(index), item.category, item.question)
    console.print(table)
    console.print(
        f"Would evaluate [bold]{len(items)}[/bold] questions with k=[bold]{top_k}[/bold] "
        f"against collection [cyan]{settings.qdrant_collection}[/cyan] "
        f"using [cyan]{settings.llm_model}[/cyan] ({settings.llm_provider})."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a POSIX exit code."""
    args = _parse_args(argv)
    logging.basicConfig(level="WARNING", format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    top_k = args.k or settings.rerank_top_n
    judge_model = args.judge_model or settings.openai_model

    try:
        items = load_benchmark(args.dataset, args.limit)
    except (OSError, ValueError, KeyError) as exc:
        console.print(f"[red]Cannot load benchmark: {exc}[/red]")
        return 2

    if args.dry_run:
        return _dry_run(items, settings, top_k)

    # Answers may be generated locally (Ollama), but the judge is always OpenAI:
    # LLM-as-a-judge metrics are only meaningful with a strong, stable grader,
    # and Ragas also needs OpenAI embeddings for answer relevance.
    if not settings.openai_api_key:
        console.print(
            "[red]Ragas needs OPENAI_API_KEY for the judge model and embeddings, "
            "even when LLM_PROVIDER=ollama generates the answers.[/red]"
        )
        return 2

    console.print(
        f"[bold]Ragas benchmark[/bold]: {len(items)} questions, k={top_k}, "
        f"generator={settings.llm_model} ({settings.llm_provider}), judge={judge_model}"
    )
    try:
        samples = asyncio.run(collect_samples(items, settings, top_k))
        scores = score_samples(samples, judge_model)
    except Exception as exc:
        logger.exception("Evaluation failed")
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        return 1

    print_report(samples, scores)
    export_results(samples, scores, settings, args.output_dir)

    faithfulness = scores["aggregate"].get("faithfulness")
    if faithfulness is not None and faithfulness < 0.7:
        console.print("[red]Faithfulness below the 0.70 quality gate.[/red]")
        return 4
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
