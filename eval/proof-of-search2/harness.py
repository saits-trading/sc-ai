"""End-to-end RAG pipeline eval: faithfulness, answer relevance, context recall.

Usage:
    python -m eval.proof_of_search2 --dataset fixtures/qa.jsonl

Dataset format (fixtures/qa.jsonl):
    {"question": "...", "ground_truth": "...", "relevant_ids": ["doc-1"], "tenant_id": "test-tenant"}

Metrics are computed without an external RAGAS server — inline LLM-as-judge
using the same completion endpoint configured in .env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import httpx

from lcore.rag import _embed, retrieve
from lproto.schemas import QueryEnvelope

COMPLETION_URL   = os.environ["COMPLETION_URL"]
COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL", "gpt-4o-mini")
COMPLETION_API_KEY = os.environ.get("COMPLETION_API_KEY", os.environ["EMBED_API_KEY"])

TARGETS = {"faithfulness": 0.85, "answer_relevance": 0.80, "context_recall": 0.70}


class QACase(NamedTuple):
    question: str
    ground_truth: str
    relevant_ids: set[str]
    tenant_id: str


class E2EMetrics(NamedTuple):
    faithfulness: float
    answer_relevance: float
    context_recall: float


def _judge(prompt: str) -> float:
    """Ask the LLM to score 0.0–1.0. Returns the float."""
    resp = httpx.post(
        COMPLETION_URL,
        headers={"Authorization": f"Bearer {COMPLETION_API_KEY}"},
        json={
            "model": COMPLETION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=30,
    ).raise_for_status().json()
    raw = resp["choices"][0]["message"]["content"].strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.0


def _faithfulness(answer: str, contexts: list[str]) -> float:
    ctx = "\n---\n".join(contexts)
    return _judge(
        f"Score 0.0–1.0: how well is the answer supported by the context only?\n"
        f"Context:\n{ctx}\n\nAnswer:\n{answer}\n\nScore (number only):"
    )


def _answer_relevance(question: str, answer: str) -> float:
    return _judge(
        f"Score 0.0–1.0: how well does the answer address the question?\n"
        f"Question: {question}\nAnswer: {answer}\n\nScore (number only):"
    )


def _context_recall(contexts: list[str], ground_truth: str) -> float:
    ctx = "\n---\n".join(contexts)
    return _judge(
        f"Score 0.0–1.0: what fraction of the ground-truth answer is covered by the context?\n"
        f"Context:\n{ctx}\n\nGround truth:\n{ground_truth}\n\nScore (number only):"
    )


def evaluate(cases: list[QACase]) -> tuple[E2EMetrics, list[dict]]:
    faith_scores, rel_scores, recall_scores = [], [], []
    details = []

    for case in cases:
        env    = QueryEnvelope(query=case.question, tenant_id=case.tenant_id)
        result = retrieve(env)

        # Generate answer (collect full stream)
        from lcore.rag import generate
        answer = "".join(generate(case.question, result.chunks))

        faith  = _faithfulness(answer, result.chunks)
        rel    = _answer_relevance(case.question, answer)
        recall = _context_recall(result.chunks, case.ground_truth)

        faith_scores.append(faith)
        rel_scores.append(rel)
        recall_scores.append(recall)
        details.append({
            "question": case.question,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevance": rel,
            "context_recall": recall,
        })

    n = len(cases) or 1
    return (
        E2EMetrics(
            faithfulness    =sum(faith_scores)  / n,
            answer_relevance=sum(rel_scores)    / n,
            context_recall  =sum(recall_scores) / n,
        ),
        details,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("fixtures/qa.jsonl"))
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    cases = [
        QACase(
            question    =row["question"],
            ground_truth=row["ground_truth"],
            relevant_ids=set(row.get("relevant_ids", [])),
            tenant_id   =row.get("tenant_id", "eval-tenant"),
        )
        for row in map(json.loads, args.dataset.read_text().splitlines())
    ]

    metrics, _ = evaluate(cases)

    print(f"\n=== proof-of-search2 e2e ({len(cases)} cases) ===")
    print(f"  faithfulness:     {metrics.faithfulness:.3f}  target≥{TARGETS['faithfulness']}")
    print(f"  answer relevance: {metrics.answer_relevance:.3f}  target≥{TARGETS['answer_relevance']}")
    print(f"  context recall:   {metrics.context_recall:.3f}  target≥{TARGETS['context_recall']}")

    passed = (
        metrics.faithfulness     >= TARGETS["faithfulness"]
        and metrics.answer_relevance >= TARGETS["answer_relevance"]
        and metrics.context_recall   >= TARGETS["context_recall"]
    )
    print(f"\n{'PASS ✓' if passed else 'FAIL ✗'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
