# Autoencoder in a FL for Benign vs. DDoS (Multi-domain setting)

_This branch anomaly-flow-baseline is part of the [anomaly-flow](https://github.com/c2dc/anomaly-flow) work._

It is only the autoencoder in a federated learning setup. The approach proposed in the paper and available in the main branch is not activated in this branch.

**To run this baseline for Anomaly-Flow, it is required to download the full Bot-IoT dataset (NF-BoT-IoT-v2) inside** `full_datasets/`. Check it [here](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA8).

---

# Reproducing this baseline for Anomaly-Flow
1. Install this baseline:

- Tested with Python 3.9.11

```commandline
$ python -m venv venv
$ source venv\bin\activate
(venv) $ pip install --upgrade pip
(venv) $ pip install Cython  # Cython first!
(venv) $ pip install -r requirements.txt
```

2. Run the autoencoder in a federated learning setup:

```commandline
(venv) $ chmod +x run.sh
(venv) $ ./run.sh
```

