## Environment variables (.env) setup

### Create the .env file
- Copy the example and edit:
```bash
cp .env.example .env
```
- Or create it directly:
```bash
cat > .env <<'ENV'
# IBM Quantum token used by app/config/tests
IBM_Q_TOKEN=your_ibm_quantum_token_here

# IBM Quantum token used by Docker/deployment
IBM_QUANTUM_TOKEN=your_ibm_quantum_token_here

# Ensure Python resolves project modules
PYTHONPATH=.
ENV
```

Notes:
- Use the same token value for both `IBM_Q_TOKEN` and `IBM_QUANTUM_TOKEN`.
- Do not use quotes or braces; just the raw token string.
- Keep `.env` at the project root. Ensure it’s ignored by Git.

### Get your IBM Quantum token
1. Sign in to the IBM Quantum Platform: [IBM Quantum Platform](https://quantum.ibm.com/).
2. Open your Account settings (API Token) and copy your token.
3. Paste it into both `IBM_Q_TOKEN` and `IBM_QUANTUM_TOKEN` in `.env`.
4. Optional reference: [Set up channel (IBM Quantum docs)](https://docs.quantum.ibm.com/guides/setup-channel).

### Quick verify
```bash
set -a; source .env; set +a
echo "$IBM_Q_TOKEN" | sed 's/./*/g'
echo "$IBM_QUANTUM_TOKEN" | sed 's/./*/g'
```
If both lines print asterisks (non-empty), the variables are loaded.