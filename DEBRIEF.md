# Project Debrief: Hybrid Quantum-Classical Knowledge Graph Link Prediction

## Executive Summary

This proof-of-concept system demonstrates a hybrid approach combining classical machine learning and quantum computing for biomedical knowledge graph link prediction. The project focuses on predicting drug-disease treatment relationships using the Hetionet biomedical knowledge graph, specifically the Compound-treats-Disease (CtD) relationship type.

The codebase is organized into four main components: benchmarking tools for performance evaluation, classical baseline models for comparison, knowledge graph processing layers for data preparation, and experimental scripts for hyperparameter tuning and analysis.

---

## 1. Benchmarking Directory

### Purpose and Overview

The benchmarking directory provides comprehensive performance evaluation, metrics tracking, and visualization capabilities. It serves as the central hub for comparing classical and quantum model performance, tracking experiment history, and analyzing scaling behavior.

### Key Components

**Dashboard Application**

The Streamlit dashboard serves as the primary user interface for exploring results. It features three main pages: a results overview displaying model performance metrics side-by-side, a live prediction interface for testing drug-disease pairs, and an experiment history page showing performance trends over time. The dashboard displays key metrics including PR-AUC scores, accuracy, and parameter counts, with visual comparisons highlighting the quantum advantage in parameter efficiency.

**Metrics Tracking System**

The metrics tracker provides structured logging and persistence of experimental results. It supports both JSON and CSV formats, enabling easy integration with the dashboard and external analysis tools. The system tracks complete experiment configurations, timestamps, and comprehensive performance metrics. It includes utilities for comparing multiple runs and generating human-readable comparison reports that highlight differences between classical and quantum approaches.

**Empirical Scaling Analysis**

The empirical scaling module measures actual runtime performance across different knowledge graph sizes. It systematically tests classical and quantum training times at various entity counts, fits mathematical models to the observed scaling behavior, and identifies crossover points where quantum approaches become computationally advantageous. The analysis assumes quadratic scaling for classical methods and logarithmic scaling for quantum approaches, extrapolating to predict performance at larger scales.

**Theoretical Scaling Projection**

A separate scalability simulation generates theoretical scaling curves based on algorithmic complexity analysis. It visualizes the expected O(N squared) classical complexity versus O(log N) quantum complexity, demonstrating the theoretical advantage quantum methods offer as knowledge graph size increases. The visualization includes crossover point annotations and explanatory notes about current NISQ hardware limitations.

**Performance Profiling**

A lightweight profiling utility uses Python's cProfile to identify bottlenecks in the pipeline execution, helping optimize the most time-consuming operations.

### Strengths

The benchmarking infrastructure provides excellent visibility into model performance with an intuitive dashboard interface. The dual approach of empirical measurement and theoretical projection offers both practical insights and theoretical validation. The metrics tracking system enables systematic comparison across experiments, supporting iterative improvement.

### Areas for Improvement

The dashboard currently uses dummy prediction logic rather than calling actual trained models through an API. The hard negative mining functionality referenced in the knowledge graph loader remains unimplemented, limiting the sophistication of negative sampling strategies.

---

## 2. Classical Baseline Directory

### Purpose and Overview

The classical baseline directory implements standard machine learning approaches for link prediction, serving as performance benchmarks against which quantum methods are compared. It focuses on handling imbalanced datasets typical in biomedical link prediction tasks.

### Key Components

**Training Pipeline**

The main training module implements a flexible ClassicalLinkPredictor class supporting multiple model types: Logistic Regression with balanced class weights, Support Vector Machines with RBF kernels, and Random Forest classifiers. The training process includes automatic feature scaling, comprehensive metric calculation, and model persistence using joblib serialization.

The module includes specialized analysis functions for regularization path exploration, examining how L1 and L2 regularization strength affects model performance. A dedicated RBF-SVC training function implements robust cross-validation with out-of-fold predictions to prevent overfitting and provide unbiased performance estimates.

**Evaluation Utilities**

The evaluation module provides comprehensive assessment tools including confusion matrix visualization, ROC curve plotting, and precision-recall curve generation. It calculates extended metrics beyond basic accuracy, including Matthews correlation coefficient and balanced accuracy, which are particularly important for imbalanced datasets. The module includes model comparison visualizations and automated report generation.

### Strengths

The implementation demonstrates strong understanding of imbalanced learning challenges, with class weighting and PR-AUC focus throughout. The cross-validation approach ensures robust evaluation, and the multiple model types enable comprehensive baseline comparisons. The code handles edge cases gracefully, including single-class scenarios and missing values.

### Areas for Improvement

There is some code duplication between the RBF-SVC cross-validation function and similar functionality in the scripts directory. The module could benefit from ensemble methods beyond Random Forest, such as gradient boosting or stacking approaches. The regularization path analysis could be extended to other model types beyond Logistic Regression.

---

## 3. Knowledge Graph Layer Directory

### Purpose and Overview

The knowledge graph layer handles all data loading, preprocessing, and embedding generation tasks. It provides the foundation for converting raw knowledge graph triples into numerical features suitable for both classical and quantum machine learning models.

### Key Components

**Data Loading**

The loader module automatically downloads the Hetionet knowledge graph from multiple mirror locations, handling compression and format variations robustly. It provides functions to extract task-specific edges, filter by relation type, and subsample entities for proof-of-concept scalability studies. The module includes train-test splitting functionality with balanced negative sampling, creating datasets suitable for supervised learning.

The loader includes placeholder functions for advanced negative sampling strategies, including similarity-based hard negative mining and adversarial negative generation, though these remain to be fully implemented.

**Embedding Generation**

The embedder module implements a sophisticated embedding pipeline with multiple fallback strategies. When PyKEEN is available, it trains TransE embeddings optimized for knowledge graph link prediction. When PyKEEN is unavailable, it falls back to deterministic random embeddings seeded by entity identifiers, ensuring reproducibility while allowing the pipeline to function without external dependencies.

The module includes PCA-based dimensionality reduction to create quantum-friendly feature spaces, reducing high-dimensional embeddings to smaller dimensions suitable for quantum circuits. It provides multiple feature construction strategies, including concatenation, difference, and Hadamard product combinations of head and tail entity embeddings.

**Feature Engineering**

A dedicated feature engineering module provides additional encoding strategies specifically designed for quantum machine learning. It includes polynomial feature generation, multiple normalization schemes, and various combination strategies for entity pair representations. The module handles dimension matching and projection to ensure features fit quantum circuit requirements.

**Embedding Quality Analysis**

The embedder includes utilities for assessing embedding quality, including within-class versus between-class similarity analysis and t-SNE visualization for exploring embedding space structure. These tools help validate that learned representations capture meaningful biomedical relationships.

### Strengths

The robust fallback mechanisms ensure the pipeline functions even when optional dependencies are missing. The flexible feature engineering supports multiple encoding strategies, enabling experimentation with different quantum feature representations. The embedding quality analysis tools provide valuable insights into representation learning effectiveness.

### Areas for Improvement

The hard negative mining functionality remains unimplemented, limiting the sophistication of training data generation. The deterministic random embeddings, while useful for testing, do not capture actual knowledge graph structure and should be replaced with learned embeddings when possible. The module could benefit from additional embedding methods beyond TransE, such as Node2Vec or GraphSAGE, which might capture different aspects of graph structure.

---

## 4. Scripts Directory

### Purpose and Overview

The scripts directory contains a comprehensive collection of experimental and analysis utilities, supporting hyperparameter tuning, model comparison, and systematic evaluation studies. These scripts enable reproducible experimentation and systematic exploration of the model space.

### Core Pipeline Scripts

The main pipeline runner executes the complete workflow from data loading through model training to result generation. It provides extensive command-line configuration options, allowing flexible experimentation without code modification. A comprehensive RBF-SVC script implements robust training with cross-validation, addressing common pitfalls in model evaluation.

### Hyperparameter Tuning Scripts

Multiple scripts support systematic hyperparameter exploration. The hyperparameter search script implements grid search across model parameters. Separate scripts focus on quantum-specific hyperparameters, including ansatz architecture search and optimizer comparison. A regularization path script analyzes the effect of regularization strength on model performance.

### Model Comparison Scripts

Several scripts enable systematic model comparison, including direct model performance comparison, embedding strategy evaluation, and quantum backend comparison between simulators and actual hardware. These scripts generate standardized reports facilitating fair comparisons.

### Evaluation and Analysis Scripts

Advanced evaluation scripts include nested cross-validation for robust performance estimation, learning curve analysis for understanding training dynamics, and statistical significance testing for comparing model performance. A multi-seed experiment script ensures robustness across random initializations.

### Specialized Experiment Scripts

The directory includes scripts for exploring specific research questions, including hard negative mining experiments and embedding validation studies. A scaling study script analyzes performance across different knowledge graph sizes.

### Utility Scripts

A shell script provides batch benchmarking capabilities, enabling automated execution of multiple experiments. A placeholder script exists for training on IBM Quantum hardware, though it remains unimplemented.

### Strengths

The comprehensive script collection enables systematic experimentation across multiple dimensions of the model space. The modular design allows scripts to be combined and extended easily. The focus on reproducibility through configuration files and standardized outputs supports scientific rigor.

### Areas for Improvement

The RBF-SVC script is extremely large, containing over 800 lines, and would benefit from refactoring into smaller, more focused modules. There is some code duplication between scripts and the main training modules, suggesting opportunities for consolidation. The quantum hardware training script remains empty and needs implementation to enable real quantum advantage demonstrations.

---

## Overall Architecture Assessment

### Data Flow

The system follows a clear pipeline: knowledge graph data flows through the loader to extract task-specific edges, the embedder generates entity representations, feature engineering transforms these into model-ready features, and both classical and quantum models train on these features. Results flow through the metrics tracker to the dashboard for visualization.

### System Strengths

The modular architecture enables independent development and testing of components. Robust error handling and fallback mechanisms ensure the system functions even with missing dependencies. Comprehensive logging and metrics tracking support scientific reproducibility. The flexible configuration system allows extensive experimentation without code changes.

### Key Challenges and Opportunities

Code duplication exists between the scripts directory and main modules, particularly around RBF-SVC training. Consolidating this logic would improve maintainability. The hard negative mining functionality, referenced in multiple places, needs implementation to improve training data quality. The dashboard requires API integration to enable real-time predictions using trained models.

The quantum hardware integration remains incomplete, with the Heron training script empty. Implementing this would enable demonstration of actual quantum advantage on real hardware. The embedding system could be enhanced with additional methods beyond TransE to capture richer graph structure.

### Current State and Readiness

The codebase represents a production-ready proof-of-concept with solid architectural foundations. It is well-positioned for hyperparameter optimization experiments, scaling studies, and systematic model comparisons. The dashboard provides accessible visualization, and the metrics tracking supports rigorous evaluation.

The system demonstrates thoughtful design for hybrid quantum-classical machine learning applied to knowledge graph link prediction, with particular attention to the challenges of imbalanced biomedical datasets. The codebase provides a strong foundation for exploring quantum advantages in parameter efficiency and scalability.

---

## Recommendations

**Immediate Priorities**

1. Implement API integration for the dashboard to enable real-time predictions
2. Complete the hard negative mining functionality to improve training data quality
3. Refactor the large RBF-SVC script into smaller, focused modules
4. Implement the quantum hardware training script for real hardware demonstrations

**Medium-Term Enhancements**

1. Consolidate duplicate code between scripts and main modules
2. Add additional embedding methods beyond TransE
3. Implement ensemble methods in the classical baseline
4. Enhance the metrics tracker with additional analysis capabilities

**Long-Term Research Directions**

1. Explore more sophisticated quantum feature maps and ansatz architectures
2. Investigate quantum advantage at larger knowledge graph scales
3. Develop domain-specific feature engineering for biomedical applications
4. Create automated hyperparameter optimization pipelines

---

## Conclusion

This hybrid quantum-classical knowledge graph link prediction system demonstrates a well-structured approach to exploring quantum advantages in machine learning. The codebase provides comprehensive tools for experimentation, evaluation, and visualization, supporting rigorous scientific investigation of quantum machine learning capabilities. While some functionality remains to be implemented, the architectural foundation is solid and ready for continued development and research.

