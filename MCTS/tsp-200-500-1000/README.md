# TSP-200-500-1000

For solving **TSP-200-500-1000** instances with 200 nodes when you **have copied heat map files of TSP-200 into** `./heatmap`:

```bash
bash solve-200.sh
```

When planning to solve TSP instanes with 500 or 1000 nodes, you could correspondingly replace the last command with `bash solve-500.sh 8` or `bash solve-1000.sh 8` and copy heat map files into `./heatmap`. After running commands, the results could be checked in dirs `results/200`, `results/500` and `results/1000`.