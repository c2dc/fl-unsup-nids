# Generalizing intrusion detection for heterogeneous networks: A stacked-unsupervised federated learning approach

## Abstract
> The constantly evolving digital transformation imposes new requirements on our society. Aspects relating to reliance on the networking domain and the difficulty of achieving security by design pose a challenge today. As a result, data-centric and machine-learning approaches arose as feasible solutions for securing large networks. Although, in the network security domain, ML-based solutions face a challenge regarding the capability to generalize between different contexts. In other words, solutions based on specific network data usually do not perform satisfactorily on other networks. This paper describes the stacked-unsupervised federated learning (FL) approach to generalize on a cross-silo configuration for a flow-based network intrusion detection system (NIDS). The proposed approach we have examined comprises a deep autoencoder in conjunction with an energy flow classifier in an ensemble learning task. Our approach performs better than traditional local learning and naive cross-evaluation (training in one context and testing on another network data). Remarkably, the proposed approach demonstrates a sound performance in the case of non-iid data silos. In conjunction with an informative feature in an ensemble architecture for unsupervised learning, we advise that the proposed FL-based NIDS results in a feasible approach for generalization between heterogeneous networks. To the best of our knowledge, our proposal is the first successful approach to applying unsupervised FL on the problem of network intrusion detection generalization using flow-based data.

# Reproducing this work
Install the requirements to reproduce this work:

```commandline
$ python -m venv venv
$ source venv\bin\activate
(venv) $ pip install --upgrade pip
(venv) $ pip install Cython
(venv) $ pip install -r requirements.txt
```
# Content of this repository


# Cite this
```
To be defined.
```
