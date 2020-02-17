# MCTS-TSP
This is the source code for solving the Traveling Salesman Problems (TSP) using **Monte Carlo tree search (MCTS)**.

### Paper
If you want to get more details, please see our paper **Targeted sampling of enlarged neighborhood via Monte Carlo tree search for TSP**. 

### Dependencies

* gcc >= 4.8.5
* Computing platform : Linux

### Quick start

For solving TSP instances with 20 nodes using **MCTS**:

```bash
cd $download-dir
cd TSP-20-50-100
bash solve-20.sh 32
```

### Usage

#### Dataset

Our model are tested on two datasets respectively, **TSP-20-50-100** and **TSPLib** which could be downloaded from:

* [TSP-20-50-100-downloading-link](https://drive.google.com/file/d/1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp/view)
* [TSPLib-downloading-link](https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95)

#### Multi-threads

If solving TSP instances faster, you can make use full of CPUs. By default, we handle them based on 32 threads:

```bash
cd $download-dir
cd TSP-20-50-100
bash solve-20.sh 32
```

By the way, our multi-threading schemes only apply in **TSP-20-50-100** dataset, not **TSPLib** instances. In addition, other shells could be found respectively in `TSP-20-50-100` and `TSPLib`. 

#### Acknowledgements

* `./TSP-20-50-100/solve-20.sh` is assigned to solve [TSP-20-50-100](https://drive.google.com/file/d/1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp/view) instances with 20 nodes;
* `./TSP-20-50-100/solve-50.sh` is assigned to solve [TSP-20-50-100](https://drive.google.com/file/d/1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp/view) instances with 50 nodes;
* `./TSP-20-50-100/solve-100.sh` is assigned to solve [TSP-20-50-100](https://drive.google.com/file/d/1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp/view) instances with 100 nodes;
* `./TSPLib/solve-tsplib.sh` is assigned to solve [TSPLib](https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95) instances.