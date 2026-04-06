# Graph-Based Movie Recommendation Using a User–Movie Bipartite Graph

**CSCI 3834 — Machine Learning with Graphs | Saint Mary's University | Winter 2026**  
**Mirza Baig · Tooba Javed**

---

## Overview

This project builds a graph-based movie recommendation system on the 
[MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) 
dataset. Movie recommendation is framed as a **link prediction** problem 
on a user–movie bipartite graph, where an edge exists between a user and 
a movie if the user rated it 4 stars or above.

Four models are implemented and compared:

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity Baseline | 0.2417 | 0.1850 | 0.2531 |
| Matrix Factorization (SVD) | 0.0417 | 0.0557 | 0.1160 |
| GraphSAGE | 0.1917 | 0.1366 | 0.2204 |
| **LightGCN** | **0.3083** | **0.2445** | **0.3090** |

LightGCN outperforms all baselines across every metric. Matrix 
factorization collapses to flat Recall (0.0557 at all K values) under 
binary interactions and chronological splitting.

---

## Dataset

- **Source:** MovieLens Latest Small — GroupLens Research Group
- **Size:** 100,836 ratings · 610 users · 9,742 movies
- **Graph edges:** ratings ≥ 4 only (positive interactions)
- **After filtering:** 48,580 positive interactions

---

## Methodology

### Graph Construction
Users and movies are represented as nodes in a bipartite graph 
G = (U ∪ M, E). An edge (u, m) is added when user u rated movie m 
with a score of 4 or above. Built with NetworkX.

### Data Preparation
- **Chronological split:** sorted by timestamp → 70% train / 10% val / 20% test
- **Cold start fix:** users or movies unseen in training moved into training set
- **Negative sampling:** each positive in val/test paired with unseen movies to form ranking candidate pools
  - Validation: 19,278 candidate pairs
  - Test: 23,256 candidate pairs

### Models

**Popularity Baseline** — ranks movies by number of training interactions. Same list for every user. Non-personalized lower bound.

**Matrix Factorization (SVD)** — learns user and item latent vectors via the [Surprise](https://surpriselib.com/) library. Trained on binary interactions.

**GraphSAGE** — two-layer mean neighborhood aggregation with learnable node embeddings. Trained with binary cross-entropy loss.  
Best config: lr=1e-3, dim=128, dropout=0.1, neg_ratio=3, 220 epochs.

**LightGCN** — removes nonlinear activations from graph convolution, keeping only neighborhood aggregation averaged across layers. Trained with BPR loss (directly optimizes ranking). Implemented with PyTorch Geometric (`LGConv`).  
Best config: lr=1e-3, dim=128, 3 layers, neg_ratio=3, reg=1e-4, 220 epochs.

Both GNN models were tuned via a learning rate sweep over 
{1e-4, 5e-4, 1e-3, 3e-3} followed by a hyperparameter search, 
with the best configuration selected by validation HR@10.

### Evaluation Metrics
All models evaluated at K ∈ {1, 5, 10, 15, 20}:
- **Precision@K** — fraction of top-K recommendations that are relevant
- **Recall@K** — fraction of all relevant items retrieved in top K
- **NDCG@K** — position-aware ranking quality, normalized against ideal ordering

---

## Key Findings

- LightGCN achieves the highest Recall and NDCG at every value of K ≥ 5, reaching Recall@20 = 0.4086
- GraphSAGE uses the same graph as LightGCN but underperforms the popularity baseline at K=10 — the training objective (cross-entropy vs. BPR) matters as much as graph access
- Matrix factorization produces flat Recall across all K values, indicating it ranks one item confidently per user and assigns near-uniform scores to everything else
- The popularity baseline is competitive due to temporal alignment with the chronological test split, not genuine personalization

---

## Requirements
```bash
pip install torch torch-geometric pandas numpy networkx scikit-surprise matplotlib
```

---

## Report

The full project report (ACM SIGCONF format) is included in the repository as `CSCI3834_FinalReport_BaigJaved.pdf`.
