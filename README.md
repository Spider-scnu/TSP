# AAAI2021

This reposity is the source code for solving the **Traveling Salesman Problems (TSP)** using **Monte Carlo tree search (MCTS)** assisted by **Graph Convolutional Network with attention mechanism (Att-GraphConvNet)**.

### Paper

* If you want to get more details, please see our paper **["Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances" by Zhang-Hua Fu, Kai-Bin Qiu and Hongyuan Zha][https://arxiv.org/abs/2012.10658]**, which is accepted by **AAAI2021**. 

### Dependencies

* Needed libraries for the Python programming language:
  * pytorch == 1.0.1.post2
  * tensorboardX
  * tensorboard
  * numpy
  * pandas
  * scikit-learn
  * multiprocessing
  * matplotlib
  * seaborn
  * scipy
  * pyconcorde
* gcc >= 4.8.5
* CUDA = 8.0
* Computing platform : Linux

### Configuration

* If you want to run our MCTS programs, you need to install [CUDA-8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive).
* After install CUDA-8.0, we need to configure its environment variables, which follow the steps bellow:

  * **First**, add **environment variables** in .bashrc

     * `gedit ~/.bashrc`
  * then add the following two lines of statements **at the end of the file which is opened above**:
     * `export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}`
     * `export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`
  * **Secondly**, set **environment variables** and dynamic link library
     * `sudo gedit /etc/profile`
  * then add the following statement **at the end**:
     * `export PATH=/usr/local/cuda/bin:$PATH`
  * **After that**, create link file
       * `sudo gedit /etc/ld.so.conf.d/cuda.conf`
  * then add the following statement:
     * `/usr/local/cuda/lib64`
  * **Finally**, run the command to make the file work:
     * `sudo ldconfig`


### Dataset

* ***Trainset:*** **Att-GraphConvNet** is trained on two datasets respectively, **TSP20-dataset** and **TSP50-dataset** which could be downloaded from:

  * [TSP-20-trainset-downloading-link](https://drive.google.com/open?id=1lmQh1SYFlcaEcvWdKZBs30GyYL-m21nb)
  * [TSP-50-trainset-downloading-link](https://drive.google.com/open?id=1VObdGvYa4k_QfrLPpYIO-tnKU431yRap)

  After decompressing trainsets, you can remove them into directories `./Att-GraphConvNet/data`.

* ***Testset:*** Our metdod is tested on some datasets respectively, **TSP-20-50-100**, **TSP-200-500-100** and **TSP-10000** which could be downloaded from:
  * [TSP-20-50-100-testset-downloading-link](https://drive.google.com/open?id=1lmQh1SYFlcaEcvWdKZBs30GyYL-m21nb)
  * [TSP-200-500-1000-testset-downloading-link](https://drive.google.com/open?id=10vIDikHjvJ4WjpU3VXrIshhl6iVwohIh)
  * [TSP-10000-testset-downloading-link](https://drive.google.com/open?id=1u0jvUSbU-cO0oXOt_JyyXElUtE9uWvNg)

  After decompressing datasets, you can copy them into directories respectively, `./MCTS/tsp-20-50-100`, `./MCTS/tsp-200-500-1000` and `./MCTS/tsp-10000`. Besides, you can copy them into directories `./Att-GraphConvNet/data`.

* ***Heatmap:*** Our team also published heat-map files and at the same time reader can download them from:

  * [TSP-20-50-100-heatmap-downloading-link](https://drive.google.com/open?id=1ApYBCWC-6YSH2dShHjPNwVJ7v84NuQPa)
  * [TSP-200-500-1000-heatmap-downloading-link](https://drive.google.com/open?id=1HUp-IDM077Xx11U8fJxmPGPJNnLwYxbT)
  * [TSP-10000-heatmap-downloading-link](https://drive.google.com/open?id=1X343yGbhJ5ytErAuTCQil1AZazTzm5u_)

  After decompressing heat-map files, you can copy them into directories respectively, `./MCTS/tsp-20-50-100/heatmap`, `./MCTS/tsp-200-500-1000/heatmap` and `./MCTS/tsp-10000/heatmap`. 

### Usage

Our method is made up of **Att-GraphConvNet** and **MCTS**. In our paper, **Att-GraphConvNet** is used to generate probabilistic heat maps which assist **MCTS** to solve **TSP**. 

* First, you can run `train-20.ipynb` to train **Att-GraphConvNet** based on **TSP-20-trainset**. If want to train models based on your own dataset,  you just need to **modify the path of dataset** in `./Att-GraphConvNet/configs/tsp20.json`. By the way, you can run `test-20-50-100.ipynb` to generate heat maps for TSP20 using trained models which are released on [TSP-models-downloading-link](https://drive.google.com/open?id=1CXckcsThmJQNfhPGvJJ-oRhvo_vVp1d4). Heat map files would be stored in directory `./Att-GraphConvNet/results/heatmap/tsp20`.  
* After generating heat maps, you can solve TSP instances with 20 nodes using **MCTS** with **single GPU**:

```bash
cd $download-dir 
cp -r $testset-dir ./MCTS/tsp-20-50-100
cp -r ./Att-GraphConvNet/results/heatmap/tsp20 ./MCTS/tsp-20-50-100/heatmap
cd ./MCTS/tsp-20-50-100
bash generate_lib.sh
bash solve-20.sh
```

### Acknowledgements

* ***Models:*** Our team also released **Att-GraphConvNet** models which are downloaded from: [TSP-models-downloading-link](https://drive.google.com/open?id=1CXckcsThmJQNfhPGvJJ-oRhvo_vVp1d4)

### Reference

* **Taillard & Helsgaun, 2019** : [LKH3](<http://akira.ruc.dk/~keld/research/LKH-3/>)
* **Concorde** : [pyconcorde](<https://github.com/jvkersch/pyconcorde>)
* **Gurobi** : [gurobi](https://www.gurobi.com/documentation/9.0/examples/tsp_py.html)
* **Kool et al., 2019** : [Attention learn to route](<https://github.com/wouterkool/attention-learn-to-route>)
* **Joshi et al., 2019** : [Graph Convolutional Network](<https://github.com/chaitjo/graph-convnet-tsp> )
* **Deudon et al., 2018** : [Encode attend naviagte](<https://github.com/MichelDeudon/encode-attend-navigate>)

