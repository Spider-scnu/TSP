# ICML2020

This reposity is the source code for solving the **Traveling Salesman Problems (TSP)** using **Monte Carlo tree search (MCTS)** assisted by **Graph Convolutional Network with attention mechanism (Att-GraphConvNet)**.

### Paper

* If you want to get more details, please see our paper **Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances**, which is submitted to ICML2020. 

### Dependencies

* Needed libraries for the Python programming language:
  * pytorch >= 1.0.1.post2
  * numpy
  * pandas
  * tensorboardX
  * sklearn
* gcc >= 4.8.5
* Computing platform : Linux

### Dataset

Our metdod is tested on some datasets respectively, **TSP-20-50-100**, **TSP-200-500-100** and **TSP-10000** which  could be downloaded from :

* [TSP-20-50-100-dataset-downloading-link](https://drive.google.com/open?id=1lmQh1SYFlcaEcvWdKZBs30GyYL-m21nb)
* [TSP-200-500-1000-dataset-downloading-link](https://drive.google.com/open?id=10vIDikHjvJ4WjpU3VXrIshhl6iVwohIh)
* [TSP-10000-dataset-downloading-link](https://drive.google.com/open?id=1u0jvUSbU-cO0oXOt_JyyXElUtE9uWvNg)

### Usage

Our method is made up of **Att-GraphConvNet** and **MCTS**. In our paper, **Att-GraphConvNet** is used to generate probabilistic heat maps which assist **MCTS** to solve **TSP**. First, .....; After that, you can solve TSP instances with 20 nodes using **MCTS** assisted by heat map:

```bash
cd $download-dir 
cp -r ./
cd ./MCTS/tsp-20-50-100
bash solve-20.sh 32
```

