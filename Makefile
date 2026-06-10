.PHONY: install dev ingest eval test lint docker-up docker-down clean check

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r backend/requirements.txt

# ── Backend ───────────────────────────────────────────────────────────────────
dev:
	cd backend && uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# ── Data ──────────────────────────────────────────────────────────────────────
ingest:
	cd backend && python ingest.py

ingest-rebuild:
	cd backend && python ingest.py --rebuild-rag

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	cd backend && python evaluate.py

eval-fast:
	cd backend && python evaluate.py --no-ragas

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend:
	cd frontend && npm install && npm start

# ── Pre-flight check (run before starting) ───────────────────────────────────
check:
	@echo "Checking Python syntax..."
	@cd backend && python3 -c "import ast,os; [ast.parse(open(os.path.join(r,f)).read()) for r,_,fs in os.walk('app') for f in fs if f.endswith('.py')]; print('All Python files OK')"
	@echo "Checking .env file..."
	@test -f backend/.env && echo ".env found" || (cp backend/.env.example backend/.env && echo "Created .env from example — FILL IN YOUR API KEYS")
	@echo "Checking data directory..."
	@ls backend/../data/*.xlsx 2>/dev/null && echo "Excel files found" || echo "WARNING: No .xlsx files in data/"

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f backend

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	cd backend && pytest tests/ -v --tb=short

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/data/faiss_store
