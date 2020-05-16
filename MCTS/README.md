# MCTS-TSP
This is the source code for solving the Traveling Salesman Problems (TSP) using **Monte Carlo tree search (MCTS)**.

### Quick start

For solving TSP instances with 20 nodes using **MCTS**:

```bash
cd $download-dir
cp -r $tsp20-heatmap-dir ./MCTS/tsp-20-50-100/heatmap
cp -r $tsp20-testset-dir ./MCTS/tsp-20-50-100
cd tsp-20-50-100
bash generate_lib.sh
bash solve-20.sh
```

