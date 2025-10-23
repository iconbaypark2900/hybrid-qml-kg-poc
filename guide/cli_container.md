# CLI Container Runbook

This guide shows how to build and run the minimal CLI-only container to execute:

* `scripts/rbf_svc_fixed.py` (classical baseline)
* `scripts/run_pipeline.py` (full hybrid pipeline)

It uses:

* `deployment/Dockerfile.cli`
* `requirements-cli-combined.txt`

## Prerequisites

* Docker installed and running
* Internet access (first run downloads Hetionet data)
* Project directory: `/home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc`

## Build the image

```bash
cd /home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc
docker build -f deployment/Dockerfile.cli -t hybrid-qml-kg-cli .
```

Notes:

* First build may take several minutes (downloads wheels).
* Subsequent builds are much faster due to layer cache.

## Start an interactive shell

Mount the repo so results/config persist on the host:

```bash
docker run --rm -it \
  -v /home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc:/app \
  -e PYTHONPATH=/app \
  hybrid-qml-kg-cli
```

Inside the container, your working directory is `/app`.

## Run the classical baseline

```bash
python scripts/rbf_svc_fixed.py
```

Outputs:

* Metrics printed to stdout
* Result JSON written under `results/` (e.g., `results/rbf_svc_*.json`)

## Run the full pipeline (hybrid)

```bash
python scripts/run_pipeline.py
```

What it does:

* Loads Hetionet
* Generates embeddings
* Trains classical baseline
* Trains quantum model (default VQC, simulator by default)
* Generates scaling plot (matplotlib headless)

## Configure quantum backend

Edit `config/quantum_config.yaml` on the host before running:

* Simulator (default, local): `execution_mode: simulator`
* Real hardware: set e.g.:

  * `execution_mode: heron`
  * `backend: ibm_torino` (or `ibm_brisbane`)

Provide your IBM Quantum token only if using hardware:

```bash
docker run --rm -it \
  -v /home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc:/app \
  -e PYTHONPATH=/app \
  -e IBM_QUANTUM_TOKEN=YOUR_TOKEN \
  hybrid-qml-kg-cli
```

## Persisting data

* `data/`, `models/`, `results/`, and `config/` are under `/app` in the container and bind-mounted to your host repo, so changes persist.

## Troubleshooting

* “Why is the build slow?”

  * First build downloads scientific wheels (numpy/scipy/qiskit/matplotlib). It’s normal. Subsequent builds are much faster.
* “Weird bash errors after a successful build”

  * If you pasted markdown bullets/code fences into the shell, bash will error on formatting. Paste only the command lines (no backticks or bullets).
* Proxy / SSL issues

  * Ensure outbound HTTPS is allowed; re-run `docker build`.
* Clean rebuild

  ```bash
  docker builder prune -f
  docker build -f deployment/Dockerfile.cli -t hybrid-qml-kg-cli .
  ```

## Quick reference

* Build:

  ```bash
  docker build -f deployment/Dockerfile.cli -t hybrid-qml-kg-cli .
  ```
* Shell:

  ```bash
  docker run --rm -it -v /home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc:/app -e PYTHONPATH=/app hybrid-qml-kg-cli
  ```
* Run baseline:

  ```bash
  python scripts/rbf_svc_fixed.py
  ```
* Run pipeline:

  ```bash
  python scripts/run_pipeline.py
  ```
