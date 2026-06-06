# saits-ai

RAG / AI services layer for the SAITS platform.

Handles vector search, retrieval-augmented generation, multi-tenant auth (Clerk-orgs),
and perimeter security (mTLS, DPoP). Backed exclusively by **Redis-Stack** — no external
vector database.

## Architecture

```
saits-ai/
├── lbus/            # Message bus — pub/sub routing, event fanout
├── lcore/           # Core logic — RAG pipeline, vector upsert/query
│   ├── rag/         # Retrieval chain, context assembly
│   └── vector/      # Redis-Stack vector index (HNSW)
├── lgateway/        # API gateway — mTLS termination, DPoP, Clerk-org routing
│   └── auth/        # JWT / DPoP validators, org-tenant mapping
├── lproto/          # Shared schemas, typed message envelopes
├── eval/
│   ├── proof-of-search/   # Retrieval accuracy eval harness
│   └── proof-of-search2/  # End-to-end RAG pipeline eval
└── infra/           # Compose profiles, Makefile helpers
```

## Quick start

```bash
# Core services only (Redis-Stack + lcore)
docker compose --profile core up -d

# Add gateway (mTLS, Clerk)
docker compose --profile core --profile gateway up -d

# Full stack including eval runner
docker compose --profile core --profile gateway --profile eval up -d
```

## Compose profiles

| Profile   | Services                                      |
|-----------|-----------------------------------------------|
| `core`    | redis-stack, lcore-api                        |
| `gateway` | lgateway (Caddy mTLS), DPoP middleware        |
| `eval`    | proof-of-search runner, metrics exporter      |

See `docker-compose.yml` and `infra/` for profile definitions.

## Environment

Copy `infra/.env.example` and fill in:

```
CLERK_SECRET_KEY=...       # Multi-tenant org auth
REDIS_URL=redis://redis-stack:6379
MTLS_CA_CERT=...           # PEM — required when profile=gateway
DPOP_KEYSET_PATH=...       # JWK keyset for DPoP bound tokens
```

## Eval harness

`eval/proof-of-search/` runs retrieval accuracy tests (precision@k, recall@k).
`eval/proof-of-search2/` extends to full pipeline: retrieval → generation → factuality scoring.

```bash
make eval-search   # proof-of-search
make eval-e2e      # proof-of-search2
```

---

## Design notes

- **No external vector-db** — all vector storage and HNSW index management via Redis-Stack;
  eliminates a separate persistence sidecar.
- **compose-profiles** keep CI lean — spin only `core` for unit tests, add `gateway` for
  integration, add `eval` for retrieval benchmarks.
- **lbus / lcore / lgateway / lproto boundary** — enforces strict import direction:
  `lproto` has no upward deps; `lcore` never imports `lgateway`.
- **mTLS inside `lgateway`** (Caddy mutual TLS) — not at the app layer.
- **DPoP (RFC 9449)** token binding enforced by `lgateway/auth/` for all inbound tokens.
- **Clerk-orgs multi-tenancy** — org ID injected as `X-Clerk-Org-Id`, routed to per-tenant
  Redis key namespaces inside `lcore`.
