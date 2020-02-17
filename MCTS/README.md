# MCTS-TSP
This is the source code for solving the Traveling Salesman Problems (TSP) using **Monte Carlo tree search (MCTS)**.

### Quick start

For solving TSP instances with 20 nodes using **MCTS**:

```bash
cd $download-dir
cp -r $tsp20-heatmap-dir ./MCTS/tsp-20-50-100/heatmap
cd tsp-20-50-100
bash solve-20.sh 32
```

### Usage

#### Multi-threads

If solving TSP instances faster, you can make use full of CPUs. By default, we handle them based on 32 threads:

```bash
cd $download-dir
cp -r $tsp20-heatmap-dir ./MCTS/tsp-20-50-100/heatmap
cd tsp-20-50-100
bash solve-20.sh 32
```

