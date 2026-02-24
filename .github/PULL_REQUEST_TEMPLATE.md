## Quantum Performance Fixes and Diagnostics

### Summary
This PR addresses critical quantum model underperformance issues and adds comprehensive diagnostics and fixes.

### Changes

#### 🔧 Quantum Performance Fixes
- **Reduced Overfitting**: Simpler feature maps (reps=1, linear entanglement) instead of complex (reps=2, full entanglement)
- **Preserved Information**: Increased dimensions from 12D to 24D, skip pre-PCA, select more features
- **Performance Optimization**: Auto-enable Nyström approximation for large datasets (>500 samples) to speed up kernel computation
- **Progress Indicators**: Added detailed logging for quantum kernel computation progress

#### 📊 New Diagnostics & Analysis
- `WHY_QUANTUM_UNDERPERFORMS.md` - Comprehensive root cause analysis
- `QUANTUM_FIXES_APPLIED.md` - Detailed fix documentation
- `QUANTUM_KERNEL_COMPUTATION.md` - Performance optimization guide

#### 🛠️ New Tools & Scripts
- `scripts/explore_parameters.py` - Systematic parameter exploration for classical and quantum models
- `scripts/find_best_quantum_config.py` - Find optimal quantum configurations using actual quantum kernels
- `scripts/fix_quantum_performance.py` - Test fixed quantum configurations
- `run_quantum_fixed.sh` - Run full pipeline with quantum fixes applied

#### 🔄 Code Improvements
- Updated `qml_trainer.py` to respect `entanglement` parameter (was hardcoded)
- Added progress logging for kernel computation
- Auto-enable Nyström approximation for large datasets
- Enhanced feature reduction pipeline

### Expected Improvements
- **Overfitting Gap**: Reduced from 0.39 to <0.20 (target)
- **Test PR-AUC**: Improved from 0.6081 to 0.70-0.75 (target)
- **Computation Speed**: 3-4x faster with Nyström approximation

### Testing
- [x] Tested with 24D quantum features
- [x] Verified Nyström approximation works correctly
- [x] Confirmed progress logging displays properly
- [x] Validated entanglement parameter is respected

### Related Issues
Addresses quantum model underperformance issues identified in analysis.
