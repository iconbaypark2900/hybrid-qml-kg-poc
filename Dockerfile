FROM python:3.11-slim

# Install Node.js 20 + pnpm
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pnpm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Environment: Python path + IBM Quantum runtime (mirrors .env.example) ---
# PYTHONPATH ensures project modules resolve inside the container (consistent
# with deployment/Dockerfile.cli and deployment/Dockerfile.featuremap).
# IBM Quantum vars default to empty so they can be injected at runtime via
# `docker run -e` or docker-compose without baking secrets into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    IBM_Q_TOKEN="" \
    IBM_QUANTUM_TOKEN="" \
    IBM_QUANTUM_INSTANCE="" \
    IBM_QUANTUM_CHANNEL=ibm_quantum_platform \
    IBM_BACKEND=""

# --- Python dependencies ---
COPY requirements-huggingface.txt ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] qiskit-algorithms>=0.3 && \
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

# Start FastAPI on 8000 (internal, matches dev_stack.sh / .env.example), Next.js on 7860 (public)
CMD ["sh", "-c", "uvicorn middleware.api:app --host 127.0.0.1 --port 8000 & cd frontend && node_modules/.bin/next start -p 7860"]
