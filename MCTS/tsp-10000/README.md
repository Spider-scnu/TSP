# TSP-10000

Firstly, running the following code to **build a dynamic library of GPU operations**:

```bash
bash generate_lib.sh
```

For solving **TSP-10000** instances with **10000 cities** when you have **copied heat map files of TSP-10000 into** `./heatmap`:

```bash
bash solve-tsp10000.sh
```

After running commands, the results could be checked in dirs `./results/result_10000.txt`.