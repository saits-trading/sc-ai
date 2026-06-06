# proof-of-search — Retrieval Accuracy Eval

Measures precision@k and recall@k for the lcore vector retrieval chain.

## Usage

```bash
make eval-search
# or
python -m eval.proof_of_search --top-k 5 --dataset fixtures/queries.jsonl
```

## Fixtures

Place labelled query/relevant-doc pairs in `fixtures/queries.jsonl`:

```jsonl
{"query": "...", "relevant_ids": ["doc-1", "doc-2"]}
```

## Metrics

| Metric       | Target |
|--------------|--------|
| precision@5  | ≥ 0.80 |
| recall@5     | ≥ 0.70 |
| MRR          | ≥ 0.85 |
