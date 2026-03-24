# Graph Based Movie Recommendation System
**Milestone 1: Dataset Exploration**

-   Load `ratings.csv` and `movies.csv` with pandas
-   Compute: number of unique users, movies, ratings; rating value distribution; ratings per user distribution; ratings per movie distribution
-   Plot: histogram of rating values, histogram of ratings per user (log scale helps here)
-   Note sparsity — users/movies that appear very rarely might be worth filtering

**Milestone 2: Graph Construction**

-   Create a NetworkX bipartite graph; prefix node IDs (e.g., `user_1`, `movie_123`) to avoid collisions
-   Add edges only for ratings ≥ 4 (positive interactions per your proposal)
-   Compute graph stats: node count by type, edge count, average degree, density
-   Optional lightweight viz: use `nx.draw` on a small subgraph (e.g., top 20 users by activity) — full graph will be too large to render meaningfully

**Milestone 3: Data Preparation** 
-   Chronological split is best for recommender systems: sort by timestamp, use earliest 70% as train, next 10% validation, last 20% test — this simulates real prediction
-   Negative sampling: for each positive (user, movie) pair in validation/test, sample ~99 movies the user hasn't rated; this is standard for ranking evaluation
-   Save processed edge lists as CSVs in `data/processed/`

 **Milestone 4: Baselines** 

-   **Popularity baseline**: rank movies by number of positive interactions in training set; recommend top-K to every user regardless of their history — simple but surprisingly competitive
-   **Matrix Factorization**: use `surprise` library (much easier than rolling your own); train SVD on training ratings, predict scores for test pairs
-   Run evaluation (Milestone 6 functions) on both so you have comparison numbers ready

**Milestone 5: GNN Model** 

-   **Recommendation**: go with **LightGCN** over GraphSAGE for this task — it's specifically designed for recommendation, simpler to implement, and likely to perform better on bipartite interaction graphs
-   Use PyTorch Geometric (`torch_geometric`); LightGCN is available as a built-in layer (`LGConv`)
-   Setup: convert your bipartite graph to a PyG `Data` object with edge_index; use BPR (Bayesian Personalized Ranking) loss — standard for implicit feedback recommendation
-   Train for ~20-50 epochs, track validation loss, save best checkpoint
-   Extract final user and movie embedding matrices for scoring

**Milestone 6: Evaluation** 

-   Implement Precision@K, Recall@K, NDCG@K in `evaluation.py` — K=10 or K=20 is typical
-   For each user in test set: score all 100 candidates (1 positive + 99 negatives), rank them, compute metrics
-   Build a clean comparison table: Popularity vs MF vs LightGCN across all three metrics
-   One plot showing performance across different K values is a nice visual

**Report + Polish** 

-   Figures: rating distribution, graph degree distribution, metric comparison table, training loss curve
-   Keep the discussion focused on _why_ the graph model does (or doesn't) outperform 




