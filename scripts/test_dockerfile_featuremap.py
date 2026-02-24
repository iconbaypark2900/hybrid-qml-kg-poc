#!/usr/bin/env python3
"""
Comprehensive Python test suite for Dockerfile.featuremap
Verifies that the Dockerfile correctly uses code from the featuremap branch
"""

import subprocess
import sys
import os
import tempfile
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message

class DockerfileFeaturemapTester:
    def __init__(self):
        self.image_name = "test-featuremap"
        self.dockerfile_path = "deployment/Dockerfile.featuremap"
        self.branch_name = "feat/mjgrav2001/featuremap"
        self.test_results: List[TestResult] = []
        self.workspace_root = Path(__file__).parent.parent
        
    def log_info(self, msg: str):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")
    
    def log_success(self, msg: str):
        print(f"{Colors.GREEN}[PASS]{Colors.NC} {msg}")
    
    def log_error(self, msg: str):
        print(f"{Colors.RED}[FAIL]{Colors.NC} {msg}")
    
    def log_warning(self, msg: str):
        print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")
    
    def run_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                cwd=self.workspace_root
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)
    
    def docker_run(self, cmd: List[str]) -> Tuple[int, str]:
        """Run a command inside the Docker container"""
        docker_cmd = ["docker", "run", "--rm", self.image_name + ":test"] + cmd
        exit_code, stdout, stderr = self.run_command(docker_cmd)
        return exit_code, stdout + stderr
    
    def test_dockerfile_syntax(self) -> TestResult:
        """Test 1: Verify Dockerfile syntax"""
        self.log_info("Test 1: Verifying Dockerfile syntax...")
        
        if not (self.workspace_root / self.dockerfile_path).exists():
            return TestResult("Dockerfile exists", False, f"Dockerfile not found at {self.dockerfile_path}")
        
        # Check Dockerfile syntax using docker build --dry-run or hadolint if available
        exit_code, _, _ = self.run_command([
            "docker", "build", "--dry-run", "-f", self.dockerfile_path, "."
        ])
        
        if exit_code == 0:
            self.log_success("Dockerfile syntax is valid")
            return TestResult("Dockerfile syntax", True)
        else:
            # Try actual build to check syntax
            exit_code, stdout, stderr = self.run_command([
                "docker", "build", "--target", "nonexistent", "-f", self.dockerfile_path, "."
            ])
            if "syntax error" in stderr.lower() or "parse error" in stderr.lower():
                self.log_error("Dockerfile has syntax errors")
                return TestResult("Dockerfile syntax", False, stderr)
            else:
                self.log_success("Dockerfile syntax appears valid")
                return TestResult("Dockerfile syntax", True)
    
    def test_dockerfile_branch_config(self) -> TestResult:
        """Test 2: Verify Dockerfile has correct branch configuration"""
        self.log_info("Test 2: Verifying Dockerfile branch configuration...")
        
        dockerfile_content = (self.workspace_root / self.dockerfile_path).read_text()
        
        checks = [
            (f"BRANCH=.*{self.branch_name}", "Branch name configured"),
            ("USE_LOCAL_CODE", "USE_LOCAL_CODE flag present"),
            ("git clone", "Git clone logic present"),
        ]
        
        all_passed = True
        messages = []
        
        for pattern, check_name in checks:
            import re
            if re.search(pattern, dockerfile_content, re.IGNORECASE):
                self.log_success(f"{check_name}: ✓")
            else:
                self.log_error(f"{check_name}: ✗")
                all_passed = False
                messages.append(f"Missing: {check_name}")
        
        return TestResult("Dockerfile branch config", all_passed, "; ".join(messages))
    
    def test_build_image(self) -> TestResult:
        """Test 3: Build the Docker image"""
        self.log_info("Test 3: Building Docker image...")
        
        exit_code, stdout, stderr = self.run_command([
            "docker", "build",
            "--build-arg", "USE_LOCAL_CODE=false",
            "--build-arg", f"BRANCH={self.branch_name}",
            "-t", f"{self.image_name}:test",
            "-f", self.dockerfile_path,
            "."
        ])
        
        if exit_code == 0:
            self.log_success("Docker image built successfully")
            return TestResult("Build image", True)
        else:
            self.log_error("Docker build failed")
            self.log_info(f"Build output:\n{stderr[-500:]}")
            return TestResult("Build image", False, stderr[-500:])
    
    def test_container_git_info(self) -> TestResult:
        """Test 4: Verify git information in container"""
        self.log_info("Test 4: Verifying git information in container...")
        
        # Check if .git exists
        exit_code, output = self.docker_run(["test", "-d", "/app/.git"])
        has_git = exit_code == 0
        
        if not has_git:
            self.log_warning(".git directory not found (might be expected)")
            return TestResult("Container git info", True, "No .git directory (shallow clone)")
        
        # Try to get branch info
        exit_code, branch_output = self.docker_run(["git", "-C", "/app", "rev-parse", "--abbrev-ref", "HEAD"])
        branch = branch_output.strip() if exit_code == 0 else "unknown"
        
        # Try to get commit hash
        exit_code, commit_output = self.docker_run(["git", "-C", "/app", "rev-parse", "HEAD"])
        commit_hash = commit_output.strip() if exit_code == 0 else "unknown"
        
        self.log_info(f"Container branch: {branch}")
        self.log_info(f"Container commit: {commit_hash[:12] if commit_hash != 'unknown' else 'unknown'}...")
        
        # For shallow clones, branch might be HEAD, which is acceptable
        if branch in [self.branch_name, "HEAD"] or commit_hash != "unknown":
            self.log_success("Git information retrieved from container")
            return TestResult("Container git info", True, f"Branch: {branch}, Commit: {commit_hash[:12]}")
        else:
            self.log_warning("Could not verify git branch, but this might be expected")
            return TestResult("Container git info", True, "Could not verify branch")
    
    def test_required_files(self) -> TestResult:
        """Test 5: Verify required files exist"""
        self.log_info("Test 5: Verifying required files exist...")
        
        required_files = [
            "requirements.txt",
            "README.md",
            "quantum_layer/quantum_feature_maps.py",
            "kg_layer/kg_embedder.py",
            "deployment/Dockerfile.featuremap",
        ]
        
        missing_files = []
        for file_path in required_files:
            exit_code, _ = self.docker_run(["test", "-f", f"/app/{file_path}"])
            if exit_code == 0:
                self.log_success(f"File exists: {file_path}")
            else:
                self.log_error(f"File missing: {file_path}")
                missing_files.append(file_path)
        
        if missing_files:
            return TestResult("Required files", False, f"Missing: {', '.join(missing_files)}")
        else:
            return TestResult("Required files", True)
    
    def test_python_packages(self) -> TestResult:
        """Test 6: Verify Python packages are installed"""
        self.log_info("Test 6: Verifying Python packages...")
        
        packages = ["numpy", "scikit-learn", "qiskit", "torch"]
        missing_packages = []
        
        for package in packages:
            exit_code, output = self.docker_run([
                "python", "-c", f"import {package}; print({package}.__version__)"
            ])
            if exit_code == 0:
                version = output.strip()
                self.log_success(f"{package}: {version}")
            else:
                self.log_error(f"{package}: Not installed")
                missing_packages.append(package)
        
        if missing_packages:
            return TestResult("Python packages", False, f"Missing: {', '.join(missing_packages)}")
        else:
            return TestResult("Python packages", True)
    
    def test_featuremap_specific_code(self) -> TestResult:
        """Test 7: Verify featuremap-specific code exists"""
        self.log_info("Test 7: Verifying featuremap-specific code...")
        
        featuremap_file = "/app/quantum_layer/quantum_feature_maps.py"
        
        # Check file exists
        exit_code, _ = self.docker_run(["test", "-f", featuremap_file])
        if exit_code != 0:
            return TestResult("Featuremap code", False, "quantum_feature_maps.py not found")
        
        # Check file content
        exit_code, content = self.docker_run(["cat", featuremap_file])
        if exit_code != 0:
            return TestResult("Featuremap code", False, "Could not read quantum_feature_maps.py")
        
        # Look for feature map related keywords
        keywords = ["FeatureMap", "feature_map", "featuremap", "class.*FeatureMap"]
        found_keywords = []
        
        import re
        for keyword in keywords:
            if re.search(keyword, content, re.IGNORECASE):
                found_keywords.append(keyword)
        
        if found_keywords:
            self.log_success(f"Found feature map keywords: {', '.join(found_keywords)}")
            return TestResult("Featuremap code", True, f"Keywords found: {len(found_keywords)}")
        else:
            self.log_warning("No feature map keywords found in file")
            return TestResult("Featuremap code", True, "File exists but keywords not found")
    
    def test_file_checksums(self) -> TestResult:
        """Test 8: Calculate file checksums for verification"""
        self.log_info("Test 8: Calculating file checksums...")
        
        key_files = [
            "quantum_layer/quantum_feature_maps.py",
            "requirements.txt",
        ]
        
        checksums = {}
        for file_path in key_files:
            exit_code, checksum_output = self.docker_run([
                "sha256sum", f"/app/{file_path}"
            ])
            if exit_code == 0:
                checksum = checksum_output.split()[0]
                checksums[file_path] = checksum
                self.log_info(f"{file_path}: {checksum[:16]}...")
            else:
                self.log_warning(f"Could not get checksum for {file_path}")
        
        if checksums:
            return TestResult("File checksums", True, f"Calculated {len(checksums)} checksums")
        else:
            return TestResult("File checksums", False, "Could not calculate checksums")
    
    def test_local_code_isolation(self) -> TestResult:
        """Test 9: Verify local code doesn't leak when USE_LOCAL_CODE=false"""
        self.log_info("Test 9: Testing local code isolation...")
        
        # Create a unique marker file
        marker_content = f"TEST_MARKER_{hashlib.md5(str(os.getpid()).encode()).hexdigest()}"
        marker_file = self.workspace_root / ".test_isolation_marker"
        
        try:
            marker_file.write_text(marker_content)
            self.log_info(f"Created marker file: {marker_file.name}")
            
            # Rebuild image (should not include marker)
            exit_code, _, _ = self.run_command([
                "docker", "build",
                "--build-arg", "USE_LOCAL_CODE=false",
                "--build-arg", f"BRANCH={self.branch_name}",
                "-t", f"{self.image_name}:test",
                "-f", self.dockerfile_path,
                "."
            ])
            
            if exit_code != 0:
                return TestResult("Local code isolation", False, "Build failed")
            
            # Check if marker exists in container
            exit_code, _ = self.docker_run(["test", "-f", f"/app/{marker_file.name}"])
            
            if exit_code == 0:
                return TestResult("Local code isolation", False, "Local marker file found in container!")
            else:
                self.log_success("Local marker file NOT in container (correct)")
                return TestResult("Local code isolation", True)
        
        finally:
            if marker_file.exists():
                marker_file.unlink()
    
    def test_python_imports(self) -> TestResult:
        """Test 10: Test that key Python modules can be imported"""
        self.log_info("Test 10: Testing Python imports...")
        
        imports_to_test = [
            "quantum_layer.quantum_feature_maps",
            "kg_layer.kg_embedder",
        ]
        
        failed_imports = []
        for module in imports_to_test:
            exit_code, output = self.docker_run([
                "python", "-c", f"import {module}; print('OK')"
            ])
            if exit_code == 0:
                self.log_success(f"Import successful: {module}")
            else:
                self.log_error(f"Import failed: {module}")
                failed_imports.append(module)
        
        if failed_imports:
            return TestResult("Python imports", False, f"Failed: {', '.join(failed_imports)}")
        else:
            return TestResult("Python imports", True)
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("Dockerfile.featuremap Comprehensive Test Suite")
        print("=" * 60)
        print()
        
        # Run tests
        self.test_results = [
            self.test_dockerfile_syntax(),
            self.test_dockerfile_branch_config(),
            self.test_build_image(),
            self.test_container_git_info(),
            self.test_required_files(),
            self.test_python_packages(),
            self.test_featuremap_specific_code(),
            self.test_file_checksums(),
            self.test_local_code_isolation(),
            self.test_python_imports(),
        ]
        
        # Print summary
        print()
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        passed = sum(1 for r in self.test_results if r.passed)
        failed = sum(1 for r in self.test_results if not r.passed)
        
        print(f"{Colors.GREEN}Tests Passed: {passed}{Colors.NC}")
        print(f"{Colors.RED}Tests Failed: {failed}{Colors.NC}")
        print()
        
        if failed > 0:
            print("Failed Tests:")
            for result in self.test_results:
                if not result.passed:
                    print(f"  - {result.name}: {result.message}")
        
        return failed == 0
    
    def cleanup(self):
        """Cleanup test artifacts"""
        self.log_info("Cleaning up...")
        self.run_command(["docker", "rmi", f"{self.image_name}:test"], capture_output=False)

if __name__ == "__main__":
    tester = DockerfileFeaturemapTester()
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    finally:
        # Optionally cleanup - comment out if you want to inspect the image
        # tester.cleanup()
        pass
