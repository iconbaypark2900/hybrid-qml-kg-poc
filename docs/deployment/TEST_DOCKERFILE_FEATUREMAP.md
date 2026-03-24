# Dockerfile.featuremap Test Suite

This document describes the comprehensive test suite for verifying that `Dockerfile.featuremap` correctly uses code from the `feat/mjgrav2001/featuremap` branch and does not accidentally use local code.

## Overview

The Dockerfile.featuremap is designed to:
1. Clone code directly from the `feat/mjgrav2001/featuremap` branch
2. Ignore local build context when `USE_LOCAL_CODE=false` (default)
3. Fall back to local code only when explicitly requested or when git clone fails

## Test Scripts

### 1. Bash Test Script (`scripts/test_dockerfile_featuremap.sh`)

Comprehensive bash-based test suite that:
- Verifies Dockerfile exists and has correct configuration
- Tests local code isolation (ensures local files don't leak into container)
- Verifies git branch information inside container
- Checks required files exist
- Validates Python environment and packages
- Tests featuremap-specific code
- Analyzes build logs

**Usage:**
```bash
cd /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc
./scripts/test_dockerfile_featuremap.sh
```

### 2. Python Test Script (`scripts/test_dockerfile_featuremap.py`)

More detailed Python-based test suite with:
- Dockerfile syntax validation
- Branch configuration verification
- Container git information extraction
- File checksum calculation for verification
- Python import testing
- Detailed error reporting

**Usage:**
```bash
cd /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc
python3 scripts/test_dockerfile_featuremap.py
```

## Test Coverage

### Critical Tests

1. **Local Code Isolation**
   - Creates a marker file locally
   - Builds image with `USE_LOCAL_CODE=false`
   - Verifies marker file is NOT in container
   - **This is the most important test** - ensures local code doesn't leak

2. **Git Branch Verification**
   - Checks that container has git information
   - Verifies branch name or commit hash
   - Confirms code came from git, not local files

3. **Required Files**
   - Verifies key files exist: `quantum_feature_maps.py`, `requirements.txt`, etc.
   - Ensures featuremap branch code is present

4. **Python Environment**
   - Tests that required packages are installed
   - Verifies Python imports work correctly

5. **Featuremap-Specific Code**
   - Checks for feature map related code in `quantum_feature_maps.py`
   - Verifies branch-specific functionality exists

## Manual Testing Steps

### Test 1: Verify Git Clone Behavior

```bash
# Build with USE_LOCAL_CODE=false (should clone from git)
docker build \
  --build-arg USE_LOCAL_CODE=false \
  --build-arg BRANCH=feat/mjgrav2001/featuremap \
  -t test-featuremap:git \
  -f deployment/Dockerfile.featuremap \
  .

# Check build logs for git clone messages
docker build ... 2>&1 | grep -i "clone\|branch"

# Verify git info in container
docker run --rm test-featuremap:git git -C /app rev-parse --abbrev-ref HEAD
docker run --rm test-featuremap:git git -C /app log -1 --oneline
```

### Test 2: Verify Local Code Isolation

```bash
# Create a test file that should NOT appear in container
echo "LOCAL_TEST_FILE" > .test_local_file

# Build image (should NOT include .test_local_file)
docker build \
  --build-arg USE_LOCAL_CODE=false \
  --build-arg BRANCH=feat/mjgrav2001/featuremap \
  -t test-featuremap:isolated \
  -f deployment/Dockerfile.featuremap \
  .

# Verify test file is NOT in container
docker run --rm test-featuremap:isolated test -f /app/.test_local_file && echo "FAIL: Local file leaked!" || echo "PASS: Local file not in container"

# Cleanup
rm .test_local_file
```

### Test 3: Verify Featuremap Branch Code

```bash
# Build image
docker build \
  --build-arg USE_LOCAL_CODE=false \
  --build-arg BRANCH=feat/mjgrav2001/featuremap \
  -t test-featuremap:verify \
  -f deployment/Dockerfile.featuremap \
  .

# Check for featuremap-specific files
docker run --rm test-featuremap:verify ls -la /app/quantum_layer/quantum_feature_maps.py

# Check file content
docker run --rm test-featuremap:verify grep -i "feature.*map" /app/quantum_layer/quantum_feature_maps.py | head -5

# Compare with local file (if on featuremap branch)
diff <(cat quantum_layer/quantum_feature_maps.py) <(docker run --rm test-featuremap:verify cat /app/quantum_layer/quantum_feature_maps.py)
```

### Test 4: Verify Requirements Installation

```bash
# Build image
docker build \
  --build-arg USE_LOCAL_CODE=false \
  --build-arg BRANCH=feat/mjgrav2001/featuremap \
  -t test-featuremap:reqs \
  -f deployment/Dockerfile.featuremap \
  .

# Check Python packages
docker run --rm test-featuremap:reqs pip list | grep -E "qiskit|numpy|scikit-learn"

# Test imports
docker run --rm test-featuremap:reqs python -c "import qiskit; import numpy; import sklearn; print('All imports OK')"
```

## Expected Behavior

### When `USE_LOCAL_CODE=false` (default):

1. Dockerfile should attempt to clone from git
2. Build logs should show: "Attempting to clone branch: feat/mjgrav2001/featuremap"
3. If clone succeeds:
   - Local build context should be overwritten
   - Container should contain code from git branch only
   - `.git` directory may or may not be present (depending on COPY logic)
4. If clone fails:
   - Warning message should appear
   - Local build context will be used (fallback)

### When `USE_LOCAL_CODE=true`:

1. Local build context should be used
2. No git clone attempt should be made
3. Container should contain local files

## Troubleshooting

### Issue: Local code appears in container when it shouldn't

**Check:**
1. Verify `USE_LOCAL_CODE=false` is set
2. Check build logs for git clone success messages
3. Verify `.dockerignore` isn't excluding necessary files
4. Check if COPY command happens before git clone logic

### Issue: Git clone fails

**Check:**
1. Verify branch exists: `git ls-remote --heads origin feat/mjgrav2001/featuremap`
2. Check if GIT_TOKEN is needed (private repo)
3. Verify network connectivity
4. Check build logs for specific error messages

### Issue: Wrong branch code in container

**Check:**
1. Verify BRANCH build arg: `--build-arg BRANCH=feat/mjgrav2001/featuremap`
2. Check git clone command in Dockerfile
3. Verify branch name spelling
4. Check if shallow clone is causing issues

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Test Dockerfile.featuremap
  run: |
    ./scripts/test_dockerfile_featuremap.sh
    
- name: Test Dockerfile.featuremap (Python)
  run: |
    python3 scripts/test_dockerfile_featuremap.py
```

## Success Criteria

All tests should pass:
- ✅ Dockerfile syntax is valid
- ✅ Branch configuration is correct
- ✅ Image builds successfully
- ✅ Local code isolation works (marker file test)
- ✅ Git information is present in container
- ✅ Required files exist
- ✅ Python packages are installed
- ✅ Featuremap-specific code exists
- ✅ Python imports work

## Notes

- The Dockerfile uses `--single-branch --depth 1` for shallow clones (faster, less storage)
- Shallow clones may show branch as "HEAD" instead of branch name (this is expected)
- The `.git` directory may not be copied to container (check COPY commands)
- File checksums can be used to verify exact code matches between local and container
