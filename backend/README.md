# PDF Chatbot Backend

FastAPI application for PDF Retrieval-Augmented Generation (RAG) Chatbot, structured with a strict layered architecture pattern.

## Request Flow Layering Contract
Every API endpoint MUST adhere to this request path:
`Router (validation & parsing) -> Service (business logic) -> Repository (query assembly) -> Database (data engine)`

*   **Routers**: Located under `app/routers/`. They are thin controllers, validating schemas and handing tasks directly to services.
*   **Services**: Located under `app/services/`. Contain core business logic, orchestrate repositories, external APIs (OpenAI), etc.
*   **Repositories**: Located under `app/repositories/`. Perform data queries, vector operations, SQL queries.
*   **Database / Vector DB**: ChromaDB and SQL database drivers.

## API camelCase Convention
All schemas returned by the API serialize automatically to camelCase to conform to the frontend contract. Write all response models to inherit from [BaseSchema](file:///app/schemas/base.py).

## Getting Started

### Installation
1. Ensure Python 3.11+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment:
   ```bash
   cp .env.example .env
   ```

### Running Server
To start the FastAPI development server:
```bash
python -m uvicorn app.main:app --reload --port 8000
```
API Documentation will be available at: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
Health checks can be queried at: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

> [!NOTE]
> For zero-cost local development, set `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=local` — this requires Ollama installed locally with a model pulled (e.g. `ollama pull mistral`) — no API keys or costs incurred during development. Switch to `LLM_PROVIDER=openai` only for final testing/demo, per the original project plan's cost-management guidance.

