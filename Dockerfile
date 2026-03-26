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
# NEXT_PUBLIC_API_URL="" → calls go to same origin, Next.js rewrites proxy to FastAPI
RUN cd frontend && NEXT_PUBLIC_API_URL="" pnpm build

# --- Copy full project (Python source) ---
COPY . .

EXPOSE 7860

# Start FastAPI on 8000 (internal), Next.js on 7860 (public)
CMD ["sh", "-c", "uvicorn middleware.api:app --host 0.0.0.0 --port 8000 & cd frontend && node_modules/.bin/next start -p 7860"]
