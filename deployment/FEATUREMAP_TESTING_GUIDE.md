# Featuremap Dockerfile Testing Guide

## Quick Start

Run the quick test to verify Dockerfile.featuremap is working:

```bash
./scripts/quick_test_featuremap.sh
```

Run comprehensive bash tests:

```bash
./scripts/test_dockerfile_featuremap.sh
```

Run comprehensive Python tests:

```bash
python3 scripts/test_dockerfile_featuremap.py
```

## What We're Testing

The Dockerfile.featuremap should:
1. ✅ Clone code from `feat/mjgrav2001/featuremap` branch (not use local code)
2. ✅ Overwrite any local files copied during build
3. ✅ Only use local code when `USE_LOCAL_CODE=true` is explicitly set
4. ✅ Install correct Python dependencies
5. ✅ Include featuremap-specific code

## Key Test: Local Code Isolation

The most critical test verifies that local code doesn't leak into the container:

1. Creates a marker file locally (e.g., `.test_local_code_marker`)
2. Builds image with `USE_LOCAL_CODE=false`
3. Checks if marker file exists in container
4. **Expected**: Marker file should NOT be in container

This test ensures the git clone logic properly overwrites the initial COPY command.

## Understanding the Dockerfile Logic

The Dockerfile has this flow:

```
1. COPY . /app/                    # Copies local files first
2. RUN if [ USE_LOCAL_CODE != true ]; then
     git clone ... /tmp/repo        # Clone from git
     rm -rf /app/*                  # Remove local files
     cp -r /tmp/repo/* /app/        # Copy git code
   fi
```

**Important**: The COPY happens first, but then git clone overwrites everything. However, hidden files (starting with `.`) might not be fully cleaned if the `rm -rf /app/.[!.]*` pattern doesn't match all cases.

## Test Results Interpretation

### ✅ All Tests Pass
- Dockerfile is working correctly
- Code from featuremap branch is being used
- Local code isolation is working

### ⚠️ Warnings (but tests pass)
- `.git` directory not found: Expected for some build scenarios
- Branch shows as "HEAD": Expected for shallow clones (`--depth 1`)
- Git clone failed but using local: Check network/branch access

### ❌ Tests Fail
- **Marker file found in container**: Local code leaked - check Dockerfile COPY/rm logic
- **Required files missing**: Git clone may have failed or wrong branch
- **Python imports fail**: Dependencies not installed correctly

## Manual Verification Steps

### Step 1: Verify Git Clone Works

```bash
# Check if branch exists remotely
git ls-remote --heads origin feat/mjgrav2001/featuremap

# Or check fork
git ls-remote --heads https://github.com/iconbaypark2900/hybrid-qml-kg-poc.git feat/mjgrav2001/featuremap
```

### Step 2: Build and Inspect

```bash
# Build image
docker build \
  --build-arg USE_LOCAL_CODE=false \
  --build-arg BRANCH=feat/mjgrav2001/featuremap \
  -t test-featuremap \
  -f deployment/Dockerfile.featuremap \
  .

# Check build logs for git clone messages
docker build ... 2>&1 | tee build.log
grep -i "clone\|branch\|successfully" build.log
```

### Step 3: Verify Container Contents

```bash
# List files in container
docker run --rm test-featuremap ls -la /app/

# Check git info
docker run --rm test-featuremap git -C /app log -1 --oneline

# Check specific file
docker run --rm test-featuremap cat /app/quantum_layer/quantum_feature_maps.py | head -20

# Compare with local (if on featuremap branch)
diff quantum_layer/quantum_feature_maps.py <(docker run --rm test-featuremap cat /app/quantum_layer/quantum_feature_maps.py)
```

### Step 4: Test Python Environment

```bash
# Test imports
docker run --rm test-featuremap python -c "
import quantum_layer.quantum_feature_maps
import kg_layer.kg_embedder
print('All imports OK')
"

# Check package versions
docker run --rm test-featuremap pip list | grep -E "qiskit|numpy|scikit-learn"
```

## Troubleshooting

### Problem: Local code appears in container

**Solution**: 
1. Verify `USE_LOCAL_CODE=false` is set
2. Check build logs show "Successfully cloned"
3. Verify `rm -rf /app/* /app/.[!.]*` executed successfully
4. Check `.dockerignore` isn't interfering

### Problem: Git clone fails silently

**Solution**:
1. Check network connectivity
2. Verify branch name is correct: `feat/mjgrav2001/featuremap`
3. Check if repo is private (needs GIT_TOKEN)
4. Try with verbose git: modify Dockerfile to remove `2>/dev/null`

### Problem: Wrong branch code in container

**Solution**:
1. Double-check BRANCH build arg
2. Verify branch exists: `git ls-remote --heads <repo> feat/mjgrav2001/featuremap`
3. Check if shallow clone (`--depth 1`) is causing issues
4. Try without `--single-branch` flag

## CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test-dockerfile.yml
name: Test Dockerfile.featuremap

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Test Dockerfile.featuremap
        run: |
          chmod +x scripts/quick_test_featuremap.sh
          ./scripts/quick_test_featuremap.sh
          
      - name: Comprehensive tests
        run: |
          python3 scripts/test_dockerfile_featuremap.py
```

## Files Created

1. `scripts/test_dockerfile_featuremap.sh` - Comprehensive bash test suite
2. `scripts/test_dockerfile_featuremap.py` - Detailed Python test suite  
3. `scripts/quick_test_featuremap.sh` - Quick verification script
4. `deployment/TEST_DOCKERFILE_FEATUREMAP.md` - Detailed test documentation
5. `deployment/FEATUREMAP_TESTING_GUIDE.md` - This guide

## Next Steps

1. Run `./scripts/quick_test_featuremap.sh` to verify basic functionality
2. If quick test passes, run comprehensive tests
3. Review test output and address any warnings
4. Integrate tests into CI/CD pipeline
5. Regularly run tests when Dockerfile changes
