# TSP-200-500-1000

Firstly, running the following code to **build a dynamic library of GPU operations**:

````bash
bash generate_lib.sh
````

For solving **TSP-200-500-1000** instances with **200 cities** when you **have copied heat map files of TSP-200 into** `./heatmap`:

```bash
bash solve-tsp200.sh
```

When planning to solve TSP instanes with **500** or **1000 cities**, you could correspondingly replace the last command with `bash solve-tsp500.sh` or `bash solve-tsp1000.sh` and copy heat map files into `./heatmap`. After running commands, the results could be checked in dirs `results/result_200.txt`, `results/result_500.txt` and `results/result_1000.txt`.