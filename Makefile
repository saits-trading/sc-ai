.PHONY: up down eval-search eval-e2e

up:
	docker compose --profile core up -d

up-full:
	docker compose --profile core --profile gateway up -d

down:
	docker compose down

eval-search:
	docker compose --profile core --profile eval run --rm eval-runner \
		python -m eval.proof_of_search

eval-e2e:
	docker compose --profile core --profile eval run --rm eval-runner \
		python -m eval.proof_of_search2
