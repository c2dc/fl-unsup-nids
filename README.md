# Generalizing intrusion detection for heterogeneous networks: A stacked-unsupervised federated learning approach

_This branch anomaly-flow-baseline is part of the [anomaly-flow](https://github.com/c2dc/anomaly-flow) work._

---

This repository relates to our paper that describes the stacked-unsupervised federated learning (FL) approach to generalize on a cross-silo configuration for a flow-based network intrusion detection system (NIDS). The proposed approach we have looked over is a deep autoencoder in conjunction with an energy flow classifier in an ensemble learning task. 

Our approach performs better than traditional local learning and naive cross-evaluation (training in one context and testing on another network data). Remarkably, the proposed approach demonstrates a sound performance in the case of non-iid data silos. Along with an informative feature in an ensemble architecture for unsupervised learning, we advise that the proposed FL-based NIDS results in a feasible approach for generalization between heterogeneous networks.

# Reproducing this work
1. Install the requirements to reproduce this work:

- Tested with Python 3.9.11

```commandline
$ python -m venv venv
$ source venv\bin\activate
(venv) $ pip install --upgrade pip
(venv) $ pip install Cython
(venv) $ pip install -r requirements.txt
```

2. Choose one of the experiments as the full datasets* (`run_full.sh`), reduced datasets (`run_reduced.sh`), or the sampled datasets (`run_sampled.sh`). For instance, running the reduced datasets:

```commandline
(venv) $ chmod +x run_reduced.sh
(venv) $ ./run_reduced.sh
```

_* the full datasets are not part of this repository, see instructions on how to download the datasets inside the _`full_datasets`_ folder._

## Some possible configurations
- To simulate other federated learning strategies of aggregation, the changes must be made to `server.py` according to [Flower documentation](https://flower.dev/docs/strategies.html).
- To remove the EFC as part of the autoencoder, remove the argument `--with-EFC` from the shellscript files.
- Select between `just benign` or `benign and attack` threshold for the autoencoder. Edit the file `client.py`, the `test_eval` assigned to `distance_calc` method refers to both thresholds, and assigned to the comparison of `losses` to `threshold_benign` for the only benign case.

# Content of this repository
```
.
├── baselines.py	 --> calculate the baselines over sampled datasets
├── baselines_reduced.py --> calculate the baselines over reduced datasets
├── client.py		 --> the source code for federated learning clients
├── error_analysis	 --> data used for error analysis
├── Error Analysis.ipynb --> notebook with error analysis
├── full_datasets	 --> reference for downloading the full datasets
├── README.md		 --> this README
├── reduced_datasets	 --> the reduced datasets (*.csv.gz)
├── requirements.txt	 --> requirements of libraries and specific versions
├── run_full.sh		 --> to execute the proposed method over full datasets
├── run_reduced.sh	 --> to execute the proposed method over reduced datasets
├── run_sampled.sh	 --> to execute the proposed method over sampled datasets
├── sampled_datasets	 --> the sampled datasets (*.csv.gz)
├── server.py		 --> the source code for federated learning server
└── utils
    ├── generate_reduced_datasets.py	--> generate reduced datasets
    ├── load_data.py			--> code for loading datasets
    └── model.py			--> code for autoencoder
```


# Cite this
```
@article{10.1016/j.cose.2023.103106,
title = {Generalizing intrusion detection for heterogeneous networks: A stacked-unsupervised federated learning approach},
journal = {Computers \& Security},
pages = {103106},
year = {2023},
issn = {0167-4048},
doi = {https://doi.org/10.1016/j.cose.2023.103106},
url = {https://www.sciencedirect.com/science/article/pii/S0167404823000160},
author = {Gustavo {de Carvalho Bertoli} and Lourenço Alves {Pereira Junior} and Osamu Saotome and Aldri Luiz {dos Santos}},
keywords = {Network Intrusion Detection, Generalization, Unsupervised Learning, Federated Learning, Network flows}
}
```
