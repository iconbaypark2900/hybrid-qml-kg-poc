import type { Algorithm } from './types';

// 32 algorithms across 7 groups — every model the pipeline can invoke
export const ALGORITHMS: Algorithm[] = [
  // Quantum kernels
  { id: 'qsvc-pauli', name: 'QSVC (Pauli)',              group: 'Quantum kernels',          mech: 'Pauli ZZ feature map → quantum kernel → classical SVM', params: 16,    runtime: '~2m',   path: 'hybrid',    status: 'live', primary: true },
  { id: 'qsvc-z',     name: 'QSVC (Z)',                  group: 'Quantum kernels',          mech: 'Z feature map → quantum kernel',                          params: 12,    runtime: '~2m',   path: 'hybrid',    status: 'dev' },
  { id: 'pqk',        name: 'Projected Quantum Kernel',  group: 'Quantum kernels',          mech: 'data re-uploading + projected measurement',                params: 32,    runtime: '~3m',   path: 'hybrid',    status: 'dev' },
  { id: 'fqk',        name: 'Fidelity Quantum Kernel',   group: 'Quantum kernels',          mech: '|⟨ψ(x)|ψ(x′)⟩|² with hardware-efficient encoding',         params: 18,    runtime: '~3m',   path: 'hybrid',    status: 'dev' },
  // Variational quantum
  { id: 'vqc',        name: 'VQC',                       group: 'Variational quantum',      mech: 'parameterized circuit + classical optimizer (COBYLA)',     params: 24,    runtime: '~3m',   path: 'hybrid',    status: 'live' },
  { id: 'qnn',        name: 'QNN',                       group: 'Variational quantum',      mech: '4-layer variational circuit + sampler primitive',          params: 48,    runtime: '~4m',   path: 'hybrid',    status: 'dev' },
  { id: 'qaoa',       name: 'QAOA',                      group: 'Variational quantum',      mech: 'cost + mixer Hamiltonian, depth p=3',                      params: 36,    runtime: '~5m',   path: 'quantum',   status: 'live' },
  { id: 'vqe-class',  name: 'VQE-classifier',            group: 'Variational quantum',      mech: 'VQE-based ground-state distinguishability',                params: 30,    runtime: '~5m',   path: 'quantum',   status: 'live' },
  // Classical baselines
  { id: 'stacking',   name: 'Stacking ensemble',         group: 'Classical baselines',      mech: 'logistic meta-classifier over RF/GBM/MLP',                 params: 2140,  runtime: '~50s',  path: 'classical', status: 'live' },
  { id: 'extratrees', name: 'Extra Trees',               group: 'Classical baselines',      mech: '500 randomized decision trees',                            params: 18000, runtime: '~25s',  path: 'classical', status: 'fallback' },
  { id: 'logreg',     name: 'Logistic Regression',       group: 'Classical baselines',      mech: 'L2-regularized linear classifier',                         params: 128,   runtime: '~5s',   path: 'classical', status: 'live' },
  { id: 'xgboost',    name: 'XGBoost',                   group: 'Classical baselines',      mech: 'gradient-boosted trees (GBDT)',                            params: 1500,  runtime: '~30s',  path: 'classical', status: 'dev' },
  { id: 'lightgbm',   name: 'LightGBM',                  group: 'Classical baselines',      mech: 'leaf-wise gradient boosting',                              params: 1200,  runtime: '~22s',  path: 'classical', status: 'dev' },
  { id: 'svm-rbf',    name: 'SVM (RBF)',                 group: 'Classical baselines',      mech: 'RBF-kernel support vector machine',                        params: 92,    runtime: '~15s',  path: 'classical', status: 'live' },
  { id: 'mlp',        name: 'MLP',                       group: 'Classical baselines',      mech: '3-layer feedforward NN, dropout 0.2',                      params: 4096,  runtime: '~40s',  path: 'classical', status: 'dev' },
  // KG embeddings
  { id: 'rotate-lr',  name: 'RotatE → LR',               group: 'KG embeddings',            mech: 'RotatE 128D rotation embedding + logistic',                params: 384,   runtime: '~40s',  path: 'classical', status: 'live' },
  { id: 'transe-lr',  name: 'TransE → LR',               group: 'KG embeddings',            mech: 'TransE translational embedding + logistic',                params: 384,   runtime: '~35s',  path: 'classical', status: 'dev' },
  { id: 'distmult',   name: 'DistMult → LR',             group: 'KG embeddings',            mech: 'DistMult bilinear scoring + logistic',                     params: 384,   runtime: '~35s',  path: 'classical', status: 'dev' },
  { id: 'complex',    name: 'ComplEx → LR',              group: 'KG embeddings',            mech: 'complex-valued embedding scoring',                         params: 768,   runtime: '~50s',  path: 'classical', status: 'dev' },
  { id: 'metapath2vec', name: 'Metapath2Vec',            group: 'KG embeddings',            mech: 'metapath-guided random walks + skipgram',                  params: 256,   runtime: '~60s',  path: 'classical', status: 'dev' },
  { id: 'node2vec',   name: 'Node2Vec',                  group: 'KG embeddings',            mech: 'biased 2nd-order random walks + skipgram',                 params: 256,   runtime: '~55s',  path: 'classical', status: 'dev' },
  // Graph neural networks
  { id: 'gcn',        name: 'GCN',                       group: 'Graph neural networks',    mech: '2-layer Kipf–Welling graph convolution',                   params: 8000,  runtime: '~90s',  path: 'classical', status: 'dev' },
  { id: 'gat',        name: 'GAT',                       group: 'Graph neural networks',    mech: 'multi-head graph attention',                               params: 12000, runtime: '~120s', path: 'classical', status: 'dev' },
  { id: 'rgcn',       name: 'R-GCN',                     group: 'Graph neural networks',    mech: 'relational GCN (one matrix per metaedge)',                 params: 16000, runtime: '~150s', path: 'classical', status: 'dev' },
  { id: 'hetero-gnn', name: 'Hetero-GNN',                group: 'Graph neural networks',    mech: 'metapath-aware heterogeneous message passing',             params: 14000, runtime: '~140s', path: 'classical', status: 'dev' },
  // Hybrid quantum-classical
  { id: 'qgnn',       name: 'QGNN',                      group: 'Hybrid quantum-classical', mech: 'quantum graph NN with parameterized circuit',              params: 64,    runtime: '~6m',   path: 'hybrid',    status: 'dev' },
  { id: 'q-rotate',   name: 'Q-RotatE',                  group: 'Hybrid quantum-classical', mech: 'quantum-augmented KGE rotation',                            params: 192,   runtime: '~5m',   path: 'hybrid',    status: 'dev' },
  { id: 'qk-meta',    name: 'Quantum Kernel + Metapath', group: 'Hybrid quantum-classical', mech: 'metapath features → quantum kernel → SVM',                 params: 28,    runtime: '~3m',   path: 'hybrid',    status: 'live' },
  { id: 'qae',        name: 'Quantum Autoencoder',       group: 'Hybrid quantum-classical', mech: 'compressed quantum representation + classical head',       params: 56,    runtime: '~4m',   path: 'hybrid',    status: 'dev' },
  // Heuristic / metapath
  { id: 'dwpc',       name: 'DWPC (Project Rephetio)',   group: 'Heuristic / metapath',     mech: 'Degree-Weighted Path Count over Hetionet metapaths',       params: 0,     runtime: '~10s',  path: 'classical', status: 'live' },
  { id: 'rwr',        name: 'Random Walk w/ Restart',    group: 'Heuristic / metapath',     mech: 'PageRank-style propagation from compound seed',            params: 1,     runtime: '~20s',  path: 'classical', status: 'live' },
  { id: 'pcg',        name: 'PathCount Geometric',       group: 'Heuristic / metapath',     mech: 'geometric mean of metapath counts',                        params: 0,     runtime: '~8s',   path: 'classical', status: 'live' }
];
