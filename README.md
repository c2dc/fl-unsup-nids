??? from here until ???END lines may have been inserted/deleted
# Generalizing intrusion detection for heterogeneous networks: A stacked-unsupervised federated learning approach

## Abstract
> The constantly evolving digital transformation imposes new requirements on our society. Aspects relating to reliance on the networking domain and the difficulty of achieving security by design pose a challenge today. As a result, data-centric and machine-learning approaches arose as feasible solutions for securing large networks. Although, in the network security domain, ML-based solutions face a challenge regarding the capability to generalize between different contexts. In other words, solutions based on specific network data usually do not perform satisfactorily on other networks. This paper describes the stacked-unsupervised federated learning (FL) approach to generalize on a cross-silo configuration for a flow-based network intrusion detection system (NIDS). The proposed approach we have examined comprises a deep autoencoder in conjunction with an energy flow classifier in an ensemble learning task. 

> Our approach performs better than traditional local learning and naive cross-evaluation (training in one context and testing on another network data). Remarkably, the proposed approach demonstrates a sound performance in the case of non-iid data silos. In conjunction with an informative feature in an ensemble architecture for unsupervised learning, we advise that the proposed FL-based NIDS results in a feasible approach for generalization between heterogeneous networks.

# Reproducing this work
Install the requirements to reproduce this work:

- Python 3.9.11

```commandline
$ python -m venv venv
$ source venv\bin\activate
(venv) $ pip install --upgrade pip
(venv) $ pip install Cython
(venv) $ pip install -r requirements.txt
```
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
    ├── generate_reduced_datasets.py	--> generated reduced datasets
    ├── load_data.py			--> code for loading datasets
    └─── model.py			--> code for autoencoder
```

To simulate other federated learning strategies of aggreagation, the changes must be made to `server.py` according to Flower documentation.

To remove the EFC as part of the autoencoder, remove the argument `--with-EFC` from the shellscript files.

# Cite this
```
To be defined.
```
