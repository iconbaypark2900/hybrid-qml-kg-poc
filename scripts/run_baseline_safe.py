#!/usr/bin/env python
"""
Safe baseline runner - forces deterministic embeddings (no PyKEEN/PyTorch memory issues).
Usage: python scripts/run_baseline_safe.py --model LogisticRegression --max_entities 300
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only

# Force fallback before importing
import kg_layer.kg_embedder as kge
kge.PYKEEN_AVAILABLE = False

# Now run the original CLI
from classical_baseline.train_baseline import main
if __name__ == "__main__":
    main()