# Knowledge Graph Embedding for Hetionet

## Scientific Foundation

The knowledge graph embedding layer transforms Hetionet's structured biomedical knowledge into continuous vector representations suitable for machine learning. Hetionet is a heterogeneous network containing entities (compounds, diseases, genes, pathways, etc.) and typed relationships (e.g., "Compound treats Disease" (CtD), "Disease associates Gene" (DaG)) represented as triples $(h, r, t)$ where $h$ is the head entity, $r$ is the relation, and $t$ is the tail entity.

The implementation uses the **TransE** (Translation Embedding) model, which embeds entities and relations in a shared $d$-dimensional space such that valid triples satisfy the translational relationship:

$$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

where $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$ are the embedding vectors. The scoring function for a triple is:

$$f(h, r, t) = \|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_2$$

During training, TransE minimizes this distance for positive triples while maximizing it for negative samples (corrupted triples), enabling the model to learn meaningful geometric relationships in the embedding space.

## Implementation

The `HetionetEmbedder` class trains TransE embeddings using PyKEEN with default parameters: embedding dimension $d=32$, 50 training epochs, and Adam optimization. When PyKEEN is unavailable, it falls back to deterministic random embeddings seeded by entity strings for pipeline continuity.

For quantum machine learning compatibility, embeddings are reduced from 32D to 5D using **Principal Component Analysis (PCA)**:

$$\mathbf{e}_{reduced} = \text{PCA}(\mathbf{e}_{full}, n_{components}=5)$$

Link prediction features are constructed from entity pairs $(h, t)$ using three modes:
- **Difference**: $|\mathbf{h} - \mathbf{t}|$ (element-wise absolute difference)
- **Hadamard**: $\mathbf{h} \odot \mathbf{t}$ (element-wise product)
- **Both**: concatenation followed by PCA projection back to 5D

The full-graph training strategy includes all Hetionet edges involving task entities (not just the target relation type), providing richer contextual information for embedding learning.

