# Dockerfile.featuremap Testing - Quick Reference

## 🚀 Quick Start

**Run the quick test (recommended first step):**
```bash
./scripts/quick_test_featuremap.sh
```

**Run comprehensive tests:**
```bash
# Bash version
./scripts/test_dockerfile_featuremap.sh

# Python version
python3 scripts/test_dockerfile_featuremap.py
```

## 📋 What Gets Tested

The test suite verifies that `Dockerfile.featuremap`:

1. ✅ **Uses code from `feat/mjgrav2001/featuremap` branch** (not local code)
2. ✅ **Isolates local code** (marker file test ensures local files don't leak)
3. ✅ **Installs correct dependencies** (Python packages)
4. ✅ **Includes required files** (quantum_feature_maps.py, etc.)
5. ✅ **Has working Python environment** (imports work)

## 📁 Test Files Created

| File | Purpose | Runtime |
|------|---------|---------|
| `scripts/quick_test_featuremap.sh` | Fast verification | ~2-3 min |
| `scripts/test_dockerfile_featuremap.sh` | Comprehensive bash tests | ~5-10 min |
| `scripts/test_dockerfile_featuremap.py` | Detailed Python tests | ~5-10 min |

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `TEST_DOCKERFILE_FEATUREMAP.md` | Detailed test documentation |
| `FEATUREMAP_TESTING_GUIDE.md` | Testing guide with examples |
| `FEATUREMAP_DOCKERFILE_ANALYSIS.md` | Dockerfile analysis & recommendations |

## 🧪 Key Test: Local Code Isolation

The most important test creates a marker file locally and verifies it's NOT in the container:

```bash
# This test is included in all test scripts
echo "TEST_MARKER" > .test_marker
docker build --build-arg USE_LOCAL_CODE=false ...
docker run ... test -f /app/.test_marker  # Should FAIL (file shouldn't exist)
```

## ✅ Success Criteria

All tests should show:
- ✅ Marker file NOT in container (local code isolation works)
- ✅ Required files exist
- ✅ Python packages installed
- ✅ Git clone succeeded (or fallback worked)
- ✅ Featuremap code present

## 🔍 Manual Verification

If tests pass but you want to double-check:

```bash
# Build image
docker build \
  --build-arg USE_LOCAL_CODE=false \
  --build-arg BRANCH=feat/mjgrav2001/featuremap \
  -t test-featuremap \
  -f deployment/Dockerfile.featuremap \
  .

# Check git info
docker run --rm test-featuremap git -C /app log -1 --oneline

# Check specific file
docker run --rm test-featuremap cat /app/quantum_layer/quantum_feature_maps.py | head -20

# Test Python
docker run --rm test-featuremap python -c "import quantum_layer.quantum_feature_maps; print('OK')"
```

## ⚠️ Common Issues

### Issue: "Local marker file found in container"
**Meaning**: Local code leaked through  
**Fix**: Check Dockerfile COPY/rm logic, verify git clone succeeded

### Issue: "Git clone failed"
**Meaning**: Can't access branch remotely  
**Fix**: Check network, verify branch exists, try with GIT_TOKEN

### Issue: "Required files missing"
**Meaning**: Wrong branch or git clone failed  
**Fix**: Verify branch name, check build logs

## 📊 Test Output Example

```
🔍 Quick Test: Dockerfile.featuremap
======================================
✓ Created marker file: .quick_test_marker_1234567890
🔨 Building Docker image...
✓ Build successful
📋 Build log analysis:
✓ Git clone attempt detected
✓ Git clone succeeded
🧪 Test 1: Local code isolation
✓ PASS: Marker file NOT in container
🧪 Test 2: Required files
✓ PASS: quantum_feature_maps.py exists
🧪 Test 3: Git information
✓ Git info found - Branch: HEAD, Commit: abc12345
🧪 Test 4: Python environment
✓ PASS: Python imports work
======================================
✅ All quick tests passed!
```

## 🎯 Next Steps

1. **Run quick test** to verify basic functionality
2. **Review test output** for any warnings
3. **Run comprehensive tests** if quick test passes
4. **Integrate into CI/CD** for continuous verification
5. **Run tests regularly** when Dockerfile changes

## 📞 Support

If tests fail:
1. Check build logs: `/tmp/docker_build*.log`
2. Review test output for specific error messages
3. See `FEATUREMAP_TESTING_GUIDE.md` for troubleshooting
4. Check `FEATUREMAP_DOCKERFILE_ANALYSIS.md` for Dockerfile details

---

**Last Updated**: Test suite created for thorough verification of Dockerfile.featuremap branch isolation
