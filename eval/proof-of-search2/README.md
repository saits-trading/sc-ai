# proof-of-search2 — End-to-End RAG Pipeline Eval

Extends proof-of-search with generation + factuality scoring.
Retrieval → LLM generation → answer vs ground-truth comparison.

## Usage

```bash
make eval-e2e
# or
python -m eval.proof_of_search2 --dataset fixtures/qa.jsonl
```

## Fixtures

`fixtures/qa.jsonl` format:

```jsonl
{"question": "...", "ground_truth": "...", "relevant_ids": ["doc-1"]}
```

## Metrics

| Metric           | Target |
|------------------|--------|
| faithfulness     | ≥ 0.85 |
| answer relevance | ≥ 0.80 |
| context recall   | ≥ 0.70 |
