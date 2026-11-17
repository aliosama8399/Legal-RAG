.PHONY: install lint test build-ui up up-ui up-full down

install:
	uv sync --extra dev --extra serve

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

build-ui:
	cd frontend && npm install && npm run build

up:
	docker compose up -d --build

up-ui:
	docker compose --profile ui up -d --build

up-full:
	docker compose --profile ui --profile tracking --profile metrics --profile langfuse up -d --build

down:
	docker compose --profile ui --profile tracking --profile metrics --profile langfuse down

