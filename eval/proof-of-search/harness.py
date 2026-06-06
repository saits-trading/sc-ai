"""Retrieval accuracy eval: precision@k, recall@k, MRR.

Usage:
    python -m eval.proof_of_search --top-k 5 --dataset fixtures/queries.jsonl

Dataset format (fixtures/queries.jsonl):
    {"query": "...", "relevant_ids": ["doc-1", "doc-2"], "tenant_id": "test-tenant"}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

from lcore.rag import _embed
from lcore.vector import search


TARGETS = {"precision": 0.80, "recall": 0.70, "mrr": 0.85}


class QueryCase(NamedTuple):
    query: str
    relevant_ids: set[str]
    tenant_id: str


class Metrics(NamedTuple):
    precision: float
    recall: float
    mrr: float


def _reciprocal_rank(hits: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(hits, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(cases: list[QueryCase], top_k: int) -> tuple[Metrics, list[dict]]:
    precisions, recalls, rrs = [], [], []
    details = []

    for case in cases:
        vec   = _embed(case.query)
        hits  = [doc_id for doc_id, _score, _text in search(case.tenant_id, vec, top_k=top_k)]
        hit_set = set(hits)

        tp = len(hit_set & case.relevant_ids)
        p  = tp / top_k if top_k else 0.0
        r  = tp / len(case.relevant_ids) if case.relevant_ids else 0.0
        rr = _reciprocal_rank(hits, case.relevant_ids)

        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)
        details.append({"query": case.query, "precision": p, "recall": r, "rr": rr, "hits": hits})

    n = len(cases) or 1
    return (
        Metrics(
            precision=sum(precisions) / n,
            recall   =sum(recalls)    / n,
            mrr      =sum(rrs)        / n,
        ),
        details,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k",   type=int,  default=5)
    parser.add_argument("--dataset", type=Path, default=Path("fixtures/queries.jsonl"))
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    cases = [
        QueryCase(
            query       =row["query"],
            relevant_ids=set(row["relevant_ids"]),
            tenant_id   =row.get("tenant_id", "eval-tenant"),
        )
        for row in map(json.loads, args.dataset.read_text().splitlines())
    ]

    metrics, details = evaluate(cases, top_k=args.top_k)

    print(f"\n=== proof-of-search @ top-{args.top_k} ({len(cases)} queries) ===")
    print(f"  precision@{args.top_k}: {metrics.precision:.3f}  target≥{TARGETS['precision']}")
    print(f"  recall@{args.top_k}:    {metrics.recall:.3f}  target≥{TARGETS['recall']}")
    print(f"  MRR:           {metrics.mrr:.3f}  target≥{TARGETS['mrr']}")

    passed = (
        metrics.precision >= TARGETS["precision"]
        and metrics.recall    >= TARGETS["recall"]
        and metrics.mrr       >= TARGETS["mrr"]
    )
    print(f"\n{'PASS ✓' if passed else 'FAIL ✗'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
