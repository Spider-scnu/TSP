# TSP-20-50-100

Firstly, running the following code to **build a dynamic library of GPU operations**:

```bash
bash generate_lib.sh
```

For solving **TSP-20-50-100** instances with **20 cities** when you **have copied heat map files of TSP-20 into** `./heatmap`:

```bash
bash solve-tsp20.sh
```

When planning to solve TSP instanes with **50** or **100 cities**, you could correspondingly replace the last command with `bash solve-tsp50.sh` or `bash solve-tsp100.sh ` and copy heat map files into `./heatmap`. After running commands, the results could be checked in dirs `results/result_20.txt`, `results/result_50.txt` and `results/result_100.txt`.