# TSP-20-50-100

For solving **TSP-20-50-100** instances with 20 nodes when you **have copied heat map files of TSP-20 into** `./heatmap`:

```bash
bash solve-20.sh
```

When planning to solve TSP instanes with 50 or 100 nodes, you could correspondingly replace the last command with `bash solve-50.sh 32` or `bash solve-100.sh 32` and copy heat map files into `./heatmap`. After running commands, the results could be checked in dirs `results/20`, `results/50` and `results/100`.