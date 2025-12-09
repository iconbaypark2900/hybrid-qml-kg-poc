#!/bin/bash
# Quick test script for Dockerfile.featuremap
# Fast verification that featuremap branch code is being used
#
# Usage:
#   ./scripts/quick_test_featuremap.sh          # Test git clone mode (default)
#   ./scripts/quick_test_featuremap.sh local    # Test local code mode
#   ./scripts/quick_test_featuremap.sh git      # Test git clone mode

set -e

IMAGE_NAME="test-featuremap-quick"
BRANCH="feat/mjgrav2001/featuremap"

# Check command line argument for mode
MODE="${1:-git}"
if [ "$MODE" = "local" ]; then
    USE_LOCAL_CODE="true"
    echo "🔧 Mode: LOCAL CODE (using files from current branch)"
else
    USE_LOCAL_CODE="false"
    echo "🔧 Mode: GIT CLONE (will try to clone from remote)"
fi

echo "🔍 Quick Test: Dockerfile.featuremap"
echo "======================================"
echo ""

# Clean up any old images to ensure fresh build
echo "🧹 Cleaning up old test images..."
docker rmi ${IMAGE_NAME} 2>/dev/null || true
echo ""

# Create marker file with unique content
MARKER=".quick_test_marker_$(date +%s)"
MARKER_CONTENT="LOCAL_CODE_LEAKED_$(date +%s)_$$"
echo "${MARKER_CONTENT}" > "${MARKER}"
echo "✓ Created marker file: ${MARKER}"
echo "  Content: ${MARKER_CONTENT}"
echo ""

# Build image with --no-cache to ensure fresh build
echo "🔨 Building Docker image (with --no-cache)..."
docker build \
    --no-cache \
    --build-arg USE_LOCAL_CODE=${USE_LOCAL_CODE} \
    --build-arg BRANCH=${BRANCH} \
    -t ${IMAGE_NAME} \
    -f deployment/Dockerfile.featuremap \
    . 2>&1 | tee /tmp/docker_build_quick.log

BUILD_EXIT_CODE=${PIPESTATUS[0]}
if [ ${BUILD_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Build successful"
else
    echo ""
    echo "✗ Build failed (exit code: ${BUILD_EXIT_CODE})"
    exit 1
fi
echo ""

# Check build logs - filter to only actual output lines (starting with #number)
echo "📋 Build log analysis:"

# Extract only the actual build output (lines starting with #<number>)
grep -E "^#[0-9]+" /tmp/docker_build_quick.log > /tmp/docker_output_only.log 2>/dev/null || true

if grep -q "Attempting to clone branch" /tmp/docker_output_only.log; then
    echo "✓ Git clone attempt detected"
else
    echo "⚠ No git clone attempt in logs"
fi

# Check for actual failure message in output (not in command text)
if grep -q "WARNING.*Failed to clone" /tmp/docker_output_only.log; then
    echo "❌ Git clone FAILED - using local code (fallback)"
    echo "  The branch may not exist on remote repos or they may be private."
    echo "  Local code will be in the container!"
    echo ""
    echo "  Options:"
    echo "    1. Push your branch to the remote repo"
    echo "    2. Use: --build-arg USE_LOCAL_CODE=true"
    echo "    3. Provide a GIT_TOKEN for private repos"
elif grep -q "Successfully cloned from" /tmp/docker_output_only.log; then
    echo "✓ Git clone succeeded"
    # Show cleanup verification from logs
    if grep -q "Removing all local files" /tmp/docker_output_only.log; then
        echo "✓ Local file cleanup attempted"
    fi
else
    echo "⚠ Could not determine clone status"
fi
echo ""

# Test 1: Local code isolation (behavior depends on mode)
echo "🧪 Test 1: Local code isolation"
if [ "$USE_LOCAL_CODE" = "true" ]; then
    # In local mode, marker file WILL be in container (expected)
    if docker run --rm ${IMAGE_NAME} test -f "/app/${MARKER}" 2>/dev/null; then
        echo "✓ PASS: Marker file found (expected in local mode)"
    else
        echo "⚠ WARNING: Marker file not found even in local mode"
    fi
else
    # In git mode, marker file should NOT be in container
    if docker run --rm ${IMAGE_NAME} test -f "/app/${MARKER}" 2>/dev/null; then
        echo "✗ FAIL: Marker file found in container (local code leaked!)"
        echo ""
        echo "  This usually means git clone failed and fell back to local code."
        echo "  Check the build logs above for 'WARNING: Failed to clone'"
        echo ""
        echo "  To fix:"
        echo "    1. Push your branch to the remote: git push origin ${BRANCH}"
        echo "    2. Or use local mode: ./scripts/quick_test_featuremap.sh local"
        exit 1
    else
        echo "✓ PASS: Marker file NOT in container (git clone working)"
    fi
fi
echo ""

# Test 2: Required file should exist
echo "🧪 Test 2: Required files"
if docker run --rm ${IMAGE_NAME} test -f "/app/quantum_layer/quantum_feature_maps.py"; then
    echo "✓ PASS: quantum_feature_maps.py exists"
else
    echo "✗ FAIL: quantum_feature_maps.py missing"
    exit 1
fi
echo ""

# Test 3: Git info
echo "🧪 Test 3: Git information"
if docker run --rm ${IMAGE_NAME} test -d "/app/.git" 2>/dev/null; then
    BRANCH_INFO=$(docker run --rm ${IMAGE_NAME} git -C /app rev-parse --abbrev-ref HEAD 2>/dev/null || echo "HEAD")
    COMMIT=$(docker run --rm ${IMAGE_NAME} git -C /app rev-parse HEAD 2>/dev/null | cut -c1-8 || echo "unknown")
    echo "✓ Git info found - Branch: ${BRANCH_INFO}, Commit: ${COMMIT}"
else
    echo "⚠ No .git directory (might be expected)"
fi
echo ""

# Test 4: Python environment
echo "🧪 Test 4: Python environment"
if docker run --rm ${IMAGE_NAME} python -c "import qiskit; import numpy; print('OK')" > /dev/null 2>&1; then
    echo "✓ PASS: Python imports work"
else
    echo "✗ FAIL: Python imports failed"
    exit 1
fi
echo ""

# Cleanup
rm -f "${MARKER}"
docker rmi ${IMAGE_NAME} > /dev/null 2>&1 || true

echo "======================================"
echo "✅ All quick tests passed!"
echo ""
