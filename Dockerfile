FROM python:3.11-slim

# Install Node.js 20 + pnpm
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pnpm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
COPY requirements-huggingface.txt ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] && \
    pip install --no-cache-dir -r requirements-huggingface.txt

# --- Next.js build ---
COPY frontend/package.json frontend/pnpm-lock.yaml ./frontend/
RUN cd frontend && pnpm install --frozen-lockfile

COPY frontend/ ./frontend/
# Rewrites use API_ORIGIN (see frontend/next.config.mjs); default matches uvicorn in this image.
RUN cd frontend && pnpm build

# --- Copy full project (Python source) ---
COPY . .

# Public HTTP must match Fly http_service.internal_port (8080). HF Spaces overrides PORT at runtime.
ENV PORT=8080
EXPOSE 8080

# FastAPI on 127.0.0.1:8000 (Next rewrites proxy); edge listens on 0.0.0.0:$PORT for Fly proxy health checks.
CMD ["sh", "-c", "uvicorn middleware.api:app --host 127.0.0.1 --port 8000 & cd frontend && exec node_modules/.bin/next start -H 0.0.0.0 -p ${PORT:-8080}"]
