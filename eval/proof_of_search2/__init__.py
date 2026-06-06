"""End-to-end RAG eval: faithfulness, answer relevance, context recall (LLM-as-judge).

Usage:
    python -m eval.proof_of_search2 --dataset eval/proof-of-search2/fixtures/qa.jsonl

Pushes results to Prometheus Pushgateway when PUSHGATEWAY_URL is set:
    sc_ai_eval_faithfulness, sc_ai_eval_answer_relevance, sc_ai_eval_context_recall
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import httpx

from lcore.rag import _embed, generate, retrieve
from lproto.schemas import QueryEnvelope

COMPLETION_URL     = os.environ["COMPLETION_URL"]
COMPLETION_MODEL   = os.environ.get("COMPLETION_MODEL", "gpt-4o-mini")
COMPLETION_API_KEY = os.environ.get("COMPLETION_API_KEY", os.environ["EMBED_API_KEY"])
PUSHGATEWAY_URL    = os.environ.get("PUSHGATEWAY_URL", "")

TARGETS = {"faithfulness": 0.85, "answer_relevance": 0.80, "context_recall": 0.70}


class QACase(NamedTuple):
    question: str
    ground_truth: str
    tenant_id: str


class E2EMetrics(NamedTuple):
    faithfulness: float
    answer_relevance: float
    context_recall: float


def _judge(prompt: str) -> float:
    resp = httpx.post(
        COMPLETION_URL,
        headers={"Authorization": f"Bearer {COMPLETION_API_KEY}"},
        json={"model": COMPLETION_MODEL, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 10, "temperature": 0},
        timeout=30,
    ).raise_for_status().json()
    try:
        return max(0.0, min(1.0, float(resp["choices"][0]["message"]["content"].strip())))
    except (ValueError, KeyError):
        return 0.0


def evaluate(cases: list[QACase]) -> tuple[E2EMetrics, list[dict]]:
    faith_s, rel_s, recall_s = [], [], []
    details = []
    for case in cases:
        env    = QueryEnvelope(query=case.question, tenant_id=case.tenant_id)
        result = retrieve(env)
        answer = "".join(generate(case.question, result.chunks))
        ctx    = "\n---\n".join(result.chunks)
        faith  = _judge(f"Score 0.0–1.0: answer supported by context only?\nContext:\n{ctx}\nAnswer:\n{answer}\nScore:")
        rel    = _judge(f"Score 0.0–1.0: answer addresses the question?\nQuestion:{case.question}\nAnswer:{answer}\nScore:")
        recall = _judge(f"Score 0.0–1.0: fraction of ground truth covered by context?\nContext:\n{ctx}\nGround truth:\n{case.ground_truth}\nScore:")
        faith_s.append(faith); rel_s.append(rel); recall_s.append(recall)
        details.append({"question": case.question, "answer": answer,
                        "faithfulness": faith, "answer_relevance": rel, "context_recall": recall})
    n = len(cases) or 1
    return E2EMetrics(sum(faith_s)/n, sum(rel_s)/n, sum(recall_s)/n), details


def _push(metrics: E2EMetrics) -> None:
    if not PUSHGATEWAY_URL:
        return
    body = "\n".join([
        "# HELP sc_ai_eval_faithfulness RAG faithfulness (LLM-as-judge)",
        "# TYPE sc_ai_eval_faithfulness gauge",
        f"sc_ai_eval_faithfulness {metrics.faithfulness}",
        "# HELP sc_ai_eval_answer_relevance Answer relevance (LLM-as-judge)",
        "# TYPE sc_ai_eval_answer_relevance gauge",
        f"sc_ai_eval_answer_relevance {metrics.answer_relevance}",
        "# HELP sc_ai_eval_context_recall Context recall (LLM-as-judge)",
        "# TYPE sc_ai_eval_context_recall gauge",
        f"sc_ai_eval_context_recall {metrics.context_recall}",
        "",
    ])
    httpx.put(
        f"{PUSHGATEWAY_URL.rstrip('/')}/metrics/job/sc_ai_eval/instance/proof_of_search2",
        content=body, headers={"Content-Type": "text/plain"}, timeout=5,
    ).raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("eval/proof-of-search2/fixtures/qa.jsonl"))
    args = parser.parse_args()
    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    cases = [
        QACase(question=row["question"], ground_truth=row["ground_truth"],
               tenant_id=row.get("tenant_id", "eval-tenant"))
        for row in map(json.loads, args.dataset.read_text().splitlines())
    ]
    metrics, _ = evaluate(cases)
    print(f"\n=== proof-of-search2 e2e ({len(cases)} cases) ===")
    print(f"  faithfulness:     {metrics.faithfulness:.3f}  target≥{TARGETS['faithfulness']}")
    print(f"  answer relevance: {metrics.answer_relevance:.3f}  target≥{TARGETS['answer_relevance']}")
    print(f"  context recall:   {metrics.context_recall:.3f}  target≥{TARGETS['context_recall']}")
    _push(metrics)
    passed = (metrics.faithfulness     >= TARGETS["faithfulness"]
              and metrics.answer_relevance >= TARGETS["answer_relevance"]
              and metrics.context_recall   >= TARGETS["context_recall"])
    print(f"\n{'PASS ✓' if passed else 'FAIL ✗'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
