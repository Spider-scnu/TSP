# ICML2020

This reposity is the source code for solving the **Traveling Salesman Problems (TSP)** using **Monte Carlo tree search (MCTS)** assisted by **Graph Convolutional Network with attention mechanism (Att-GraphConvNet)**.

### Paper

* If you want to get more details, please see our paper **Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances**, which is submitted to ICML2020. 

### Dependencies

* Needed libraries for the Python programming language:
  * pytorch == 1.0.1.post2
  * tensorflow == 1.12.0
  * tensorboardX
  * tensorboard
  * numpy
  * pandas
  * scikit-learn
  * multiprocessing
  * matplotlib
  * seaborn
  * scipy
* gcc >= 4.8.5
* Computing platform : Linux

### Dataset

* ***Trainset:*** **Att-GraphConvNet** is trained on two datasets respectively, **TSP20-dataset** and **TSP50-dataset** which could be downloaded from:

  * [TSP-20-trainset-downloading-link](https://drive.google.com/open?id=1lmQh1SYFlcaEcvWdKZBs30GyYL-m21nb)
  * [TSP-50-trainset-downloading-link](https://drive.google.com/open?id=1lmQh1SYFlcaEcvWdKZBs30GyYL-m21nb)

  After decompressing trainsets, you can remove them into directories `./Att-GraphConvNet/data`.

* ***Testset:*** Our metdod is tested on some datasets respectively, **TSP-20-50-100**, **TSP-200-500-100** and **TSP-10000** which could be downloaded from:
  * [TSP-20-50-100-dataset-downloading-link](https://drive.google.com/open?id=1lmQh1SYFlcaEcvWdKZBs30GyYL-m21nb)
  * [TSP-200-500-1000-dataset-downloading-link](https://drive.google.com/open?id=10vIDikHjvJ4WjpU3VXrIshhl6iVwohIh)
  * [TSP-10000-dataset-downloading-link](https://drive.google.com/open?id=1u0jvUSbU-cO0oXOt_JyyXElUtE9uWvNg)

  After decompressing datasets, you can copy them into directories respectively, `./MCTS/tsp-20-50-100`, `./MCTS/tsp-200-500-1000` and `./MCTS/tsp-10000`. Besides, you can copy them into directories `./Att-GraphConvNet/data`.

* ***Heatmap:*** Our team also published heat-map files and at the same time reader can download them from:

  * [TSP-20-50-100-heatmap-downloading-link](https://drive.google.com/open?id=1rD5zhaZ-Ky09LNua2VMdVGkL5D3_G6EU)
  * [TSP-200-500-1000-heatmap-downloading-link](https://drive.google.com/open?id=1HUp-IDM077Xx11U8fJxmPGPJNnLwYxbT)
  * [TSP-10000-heatmap-downloading-link](https://drive.google.com/open?id=1X343yGbhJ5ytErAuTCQil1AZazTzm5u_)

  After decompressing heat-map files, you can copy them into directories respectively, `./MCTS/tsp-20-50-100/heatmap`, `./MCTS/tsp-200-500-1000/heatmap` and `./MCTS/tsp-10000/heatmap`. 

### Usage

Our method is made up of **Att-GraphConvNet** and **MCTS**. In our paper, **Att-GraphConvNet** is used to generate probabilistic heat maps which assist **MCTS** to solve **TSP**. First, .....; 







After generating heat maps, you can solve TSP instances with 20 nodes using **MCTS**:

```bash
cd $download-dir 
cp -r ./
cd ./MCTS/tsp-20-50-100
bash solve-20.sh 32
```

