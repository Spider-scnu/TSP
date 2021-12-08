## Att-GraphConvNet

### Usage

* **Train your own model**
  * If want to train your own model based on our datasets, you can download datasets from [TSP-20-trainset-downloading-link](https://drive.google.com/file/d/1zfk5k4mIuSu8wZqZl9Zly5P9xnqPs8Bv/view?usp=sharing) or [TSP-50-trainset-downloading-link](https://drive.google.com/open?id=1VObdGvYa4k_QfrLPpYIO-tnKU431yRap) at first. And then you just need to run the third code cell in `train-20.ipynb` to train the model.
  * If want to train your own model based on your own datasets, you must modify the configure of `./configs/tsp20.json` and also move your own dataset into `./data`. And then you just need to run the third code cell in `train-20.ipynb` to train the model.
* **Generating Heatmap files**
  * If want to generate heatmap files based on our testsets, you can download testsets from [TSP-20-50-100-testset-downloading-link](https://drive.google.com/open?id=1lmQh1SYFlcaEcvWdKZBs30GyYL-m21nb), [TSP-200-500-1000-testset-downloading-link](https://drive.google.com/open?id=10vIDikHjvJ4WjpU3VXrIshhl6iVwohIh) or [TSP-10000-testset-downloading-link](https://drive.google.com/open?id=1u0jvUSbU-cO0oXOt_JyyXElUtE9uWvNg) at first. And then you just need to run corresponding code cells in `test-20-50-100.ipynb` to generate heatmap files. After generating heatmaps, you must copy them into `./MCTS/TSP-XXX/heatmap` in order to solve TSP-instances using **MCTS**.
