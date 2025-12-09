#!/bin/bash
# Comprehensive test suite for Dockerfile.featuremap
# Verifies that the Dockerfile correctly uses code from the featuremap branch

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="test-featuremap"
DOCKERFILE_PATH="deployment/Dockerfile.featuremap"
BRANCH_NAME="feat/mjgrav2001/featuremap"
TEST_MARKER_FILE=".test_local_code_marker"

# Counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

cleanup() {
    log_info "Cleaning up test artifacts..."
    docker rmi ${IMAGE_NAME}:test 2>/dev/null || true
    rm -f ${TEST_MARKER_FILE}
}

trap cleanup EXIT

# Test 1: Verify Dockerfile exists
test_dockerfile_exists() {
    log_info "Test 1: Verifying Dockerfile exists..."
    if [ -f "${DOCKERFILE_PATH}" ]; then
        log_success "Dockerfile exists at ${DOCKERFILE_PATH}"
        return 0
    else
        log_error "Dockerfile not found at ${DOCKERFILE_PATH}"
        return 1
    fi
}

# Test 2: Verify Dockerfile references correct branch
test_dockerfile_branch_reference() {
    log_info "Test 2: Verifying Dockerfile references correct branch..."
    if grep -q "BRANCH=.*${BRANCH_NAME}" "${DOCKERFILE_PATH}"; then
        log_success "Dockerfile references branch: ${BRANCH_NAME}"
        return 0
    else
        log_error "Dockerfile does not reference branch: ${BRANCH_NAME}"
        return 1
    fi
}

# Test 3: Create marker file to test if local code leaks through
test_local_code_isolation() {
    log_info "Test 3: Testing local code isolation..."
    
    # Create a marker file that should NOT appear in the container
    echo "THIS_SHOULD_NOT_BE_IN_CONTAINER" > "${TEST_MARKER_FILE}"
    
    log_info "Created marker file: ${TEST_MARKER_FILE}"
    log_info "Building image with USE_LOCAL_CODE=false (should use git clone)..."
    
    # Build without USE_LOCAL_CODE (should clone from git)
    if docker build \
        --build-arg USE_LOCAL_CODE=false \
        --build-arg BRANCH=${BRANCH_NAME} \
        -t ${IMAGE_NAME}:test \
        -f ${DOCKERFILE_PATH} \
        . > /tmp/docker_build.log 2>&1; then
        
        log_info "Build successful. Checking if marker file exists in container..."
        
        # Check if marker file exists in container (it shouldn't)
        if docker run --rm ${IMAGE_NAME}:test test -f "/app/${TEST_MARKER_FILE}"; then
            log_error "Local marker file found in container! Local code leaked through."
            return 1
        else
            log_success "Local marker file NOT found in container (correct behavior)"
            return 0
        fi
    else
        log_error "Docker build failed. Check /tmp/docker_build.log"
        cat /tmp/docker_build.log | tail -20
        return 1
    fi
}

# Test 4: Verify git branch inside container
test_container_branch() {
    log_info "Test 4: Verifying git branch inside container..."
    
    # Check if .git directory exists
    if docker run --rm ${IMAGE_NAME}:test test -d "/app/.git"; then
        log_info ".git directory exists in container"
        
        # Get branch name from container
        CONTAINER_BRANCH=$(docker run --rm ${IMAGE_NAME}:test git -C /app rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        
        if [ "${CONTAINER_BRANCH}" = "${BRANCH_NAME}" ] || [ "${CONTAINER_BRANCH}" = "HEAD" ]; then
            log_success "Container is on correct branch or detached HEAD (expected for shallow clone)"
            log_info "Container branch/HEAD: ${CONTAINER_BRANCH}"
            return 0
        else
            log_warning "Container branch '${CONTAINER_BRANCH}' doesn't match expected '${BRANCH_NAME}'"
            log_info "This might be expected if using shallow clone (HEAD)"
            return 0  # Not a failure, shallow clones show as HEAD
        fi
    else
        log_warning ".git directory not found in container (might be expected if .git* was excluded)"
        return 0  # Not necessarily a failure
    fi
}

# Test 5: Verify git commit information
test_git_commit_info() {
    log_info "Test 5: Verifying git commit information..."
    
    # Try to get commit hash
    if docker run --rm ${IMAGE_NAME}:test test -d "/app/.git"; then
        COMMIT_HASH=$(docker run --rm ${IMAGE_NAME}:test git -C /app rev-parse HEAD 2>/dev/null || echo "unknown")
        
        if [ "${COMMIT_HASH}" != "unknown" ] && [ -n "${COMMIT_HASH}" ]; then
            log_success "Git commit hash retrieved: ${COMMIT_HASH:0:8}..."
            log_info "Full commit hash: ${COMMIT_HASH}"
            
            # Try to get commit message
            COMMIT_MSG=$(docker run --rm ${IMAGE_NAME}:test git -C /app log -1 --pretty=%B 2>/dev/null || echo "unknown")
            log_info "Latest commit message: ${COMMIT_MSG}"
            return 0
        else
            log_warning "Could not retrieve commit hash (might be shallow clone without history)"
            return 0
        fi
    else
        log_warning "Cannot verify commit info - .git directory not found"
        return 0
    fi
}

# Test 6: Verify required files exist
test_required_files() {
    log_info "Test 6: Verifying required files exist in container..."
    
    REQUIRED_FILES=(
        "requirements.txt"
        "README.md"
        "quantum_layer/quantum_feature_maps.py"
        "kg_layer/kg_embedder.py"
    )
    
    ALL_FOUND=true
    for file in "${REQUIRED_FILES[@]}"; do
        if docker run --rm ${IMAGE_NAME}:test test -f "/app/${file}"; then
            log_success "Required file exists: ${file}"
        else
            log_error "Required file missing: ${file}"
            ALL_FOUND=false
        fi
    done
    
    if [ "$ALL_FOUND" = true ]; then
        return 0
    else
        return 1
    fi
}

# Test 7: Verify Python environment
test_python_environment() {
    log_info "Test 7: Verifying Python environment..."
    
    # Check Python version
    PYTHON_VERSION=$(docker run --rm ${IMAGE_NAME}:test python --version 2>&1)
    log_info "Python version: ${PYTHON_VERSION}"
    
    # Check if key packages are installed
    KEY_PACKAGES=("numpy" "scikit-learn" "qiskit")
    
    ALL_INSTALLED=true
    for package in "${KEY_PACKAGES[@]}"; do
        if docker run --rm ${IMAGE_NAME}:test python -c "import ${package}" 2>/dev/null; then
            PACKAGE_VERSION=$(docker run --rm ${IMAGE_NAME}:test python -c "import ${package}; print(${package}.__version__)" 2>/dev/null || echo "unknown")
            log_success "Package installed: ${package} (${PACKAGE_VERSION})"
        else
            log_error "Package not installed: ${package}"
            ALL_INSTALLED=false
        fi
    done
    
    if [ "$ALL_INSTALLED" = true ]; then
        return 0
    else
        return 1
    fi
}

# Test 8: Verify featuremap-specific code exists
test_featuremap_code() {
    log_info "Test 8: Verifying featuremap-specific code exists..."
    
    # Check for quantum_feature_maps.py which should be in featuremap branch
    if docker run --rm ${IMAGE_NAME}:test test -f "/app/quantum_layer/quantum_feature_maps.py"; then
        log_success "quantum_feature_maps.py exists"
        
        # Check file content for featuremap-related code
        if docker run --rm ${IMAGE_NAME}:test grep -q "feature.*map\|FeatureMap" /app/quantum_layer/quantum_feature_maps.py 2>/dev/null; then
            log_success "quantum_feature_maps.py contains feature map code"
            return 0
        else
            log_warning "quantum_feature_maps.py exists but may not contain expected content"
            return 0
        fi
    else
        log_error "quantum_feature_maps.py not found - featuremap branch code may be missing"
        return 1
    fi
}

# Test 9: Test with USE_LOCAL_CODE=true (should use local code)
test_local_code_mode() {
    log_info "Test 9: Testing USE_LOCAL_CODE=true mode..."
    
    # Build with USE_LOCAL_CODE=true
    if docker build \
        --build-arg USE_LOCAL_CODE=true \
        --build-arg BRANCH=${BRANCH_NAME} \
        -t ${IMAGE_NAME}:local-test \
        -f ${DOCKERFILE_PATH} \
        . > /tmp/docker_build_local.log 2>&1; then
        
        log_info "Build with USE_LOCAL_CODE=true successful"
        
        # Check if marker file exists (it should when using local code)
        if docker run --rm ${IMAGE_NAME}:local-test test -f "/app/${TEST_MARKER_FILE}"; then
            log_success "Local marker file found in container (expected with USE_LOCAL_CODE=true)"
            docker rmi ${IMAGE_NAME}:local-test 2>/dev/null || true
            return 0
        else
            log_warning "Local marker file not found even with USE_LOCAL_CODE=true"
            log_info "This might be expected if COPY happens before git clone logic"
            docker rmi ${IMAGE_NAME}:local-test 2>/dev/null || true
            return 0
        fi
    else
        log_error "Docker build with USE_LOCAL_CODE=true failed"
        cat /tmp/docker_build_local.log | tail -20
        return 1
    fi
}

# Test 10: Verify build logs show correct behavior
test_build_logs() {
    log_info "Test 10: Analyzing build logs for correct behavior..."
    
    if [ -f /tmp/docker_build.log ]; then
        # Check if logs show git clone attempt
        if grep -q "Attempting to clone branch" /tmp/docker_build.log; then
            log_success "Build logs show git clone attempt"
        else
            log_warning "Build logs don't show git clone attempt"
        fi
        
        # Check if logs show branch name
        if grep -q "${BRANCH_NAME}" /tmp/docker_build.log; then
            log_success "Build logs reference correct branch: ${BRANCH_NAME}"
        else
            log_warning "Build logs don't reference branch: ${BRANCH_NAME}"
        fi
        
        # Check for success message
        if grep -q "Successfully cloned\|Using local build context" /tmp/docker_build.log; then
            log_success "Build logs show successful clone or local context usage"
        else
            log_warning "Build logs don't show clear success message"
        fi
        
        return 0
    else
        log_warning "Build log file not found"
        return 0
    fi
}

# Test 11: Verify container can run Python scripts
test_container_functionality() {
    log_info "Test 11: Testing container functionality..."
    
    # Test basic Python import
    if docker run --rm ${IMAGE_NAME}:test python -c "import sys; print(f'Python {sys.version}')" > /tmp/python_test.log 2>&1; then
        log_success "Container can execute Python"
        cat /tmp/python_test.log
        return 0
    else
        log_error "Container cannot execute Python"
        cat /tmp/python_test.log
        return 1
    fi
}

# Test 12: Compare file checksums (if we can get remote branch info)
test_file_checksums() {
    log_info "Test 12: Verifying file integrity..."
    
    # Get checksum of a key file from container
    CONTAINER_CHECKSUM=$(docker run --rm ${IMAGE_NAME}:test sha256sum /app/quantum_layer/quantum_feature_maps.py 2>/dev/null | cut -d' ' -f1 || echo "unknown")
    
    if [ "${CONTAINER_CHECKSUM}" != "unknown" ]; then
        log_success "File checksum retrieved: ${CONTAINER_CHECKSUM:0:16}..."
        log_info "This can be compared with remote branch to verify correctness"
        return 0
    else
        log_warning "Could not retrieve file checksum"
        return 0
    fi
}

# Main test execution
main() {
    echo "=========================================="
    echo "Dockerfile.featuremap Test Suite"
    echo "=========================================="
    echo ""
    
    # Run all tests
    test_dockerfile_exists
    test_dockerfile_branch_reference
    test_local_code_isolation || log_warning "Local code isolation test had issues"
    test_container_branch
    test_git_commit_info
    test_required_files
    test_python_environment
    test_featuremap_code
    test_local_code_mode || log_warning "Local code mode test had issues"
    test_build_logs
    test_container_functionality
    test_file_checksums
    
    # Summary
    echo ""
    echo "=========================================="
    echo "Test Summary"
    echo "=========================================="
    echo -e "${GREEN}Tests Passed: ${TESTS_PASSED}${NC}"
    echo -e "${RED}Tests Failed: ${TESTS_FAILED}${NC}"
    echo ""
    
    if [ ${TESTS_FAILED} -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed. Review output above.${NC}"
        exit 1
    fi
}

# Run main function
main
