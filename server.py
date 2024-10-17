import argparse
import flwr as fl
from utils import model, load_data
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# from flwr.dataset.utils.common import create_lda_partitions # latent dirichlet allocation distribution

# def centralized_eval(weights):
# 	model =
# 	test_data =
# 	parameters = weights_to_parameters(weights)
# 	params_dict = zip(model.state_dict().keys(), weights)
# 	state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
# 	model.load_state_dict(state_dict, strict=True)
# 	test_loss, test_acc, num_samples = test(model, test_data)
# 	metrics = {"centralized_acc" : test_acc, "num_samples" : num_samples}


def fit_config(rnd: int):
    config = {
        "epoch_global": str(rnd),
        "num_epochs": 1,
        "batch_size": 32
    }
    return config


def average_metrics(metrics):
    accuracies = [metric["acc"] for _, metric in metrics]
    recalls = [metric["rec"] for _, metric in metrics]
    precisions = [metric["prec"] for _, metric in metrics]
    f1s = [metric["f1"] for _, metric in metrics]
    mccs = [metric["mcc"] for _, metric in metrics]
    missrates = [metric["missrate"] for _, metric in metrics]
    fallouts = [metric["fallout"] for _, metric in metrics]
    aucs = [metric["auc"] for _, metric in metrics]

    accuracies = sum(accuracies) / len(accuracies)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)
    f1s = sum(f1s) / len(f1s)
    mccs = sum(mccs) / len(mccs)
    missrates = sum(missrates) / len(missrates)
    fallouts = sum(fallouts) / len(fallouts)
    aucs = sum(aucs) / len(aucs)

    return {"acc": accuracies, "rec": recalls, "prec": precisions, "f1": f1s, "mcc": mccs, "missrate": missrates,
            "fallout": fallouts, "auc": aucs}


def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--with_efc", required=False, action="store_true")
    parser.add_argument("--full", required=False, action="store_true")
    args = parser.parse_args()

    # initializing a generic model to get its parameters
    # reference: https://github.com/adap/flower/blob/main/tutorials/Flower-2-Strategies-in-FL-PyTorch.ipynb
    if args.full:
        sample_silo = "./full_datasets/NF-UNSW-NB15-v2.csv.gz"
    else:
        sample_silo = "./sampled_datasets/toniot_sampled.csv.gz"
    x_train, _, _, _ = load_data.load_data(sample_silo, info=False, full=args.full)
    
    if args.with_efc:
        params = model.create_model(x_train.shape[1] + 1).get_weights()  # Additional Feature if using EFC
    else:
        params = model.create_model(x_train.shape[1]).get_weights()

    del x_train

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=average_metrics,
        # eval_fn=centralized_eval,
        # on_fit_config_fn=fit_config,
        initial_parameters=fl.common.weights_to_parameters(params)

        # FedAvgM Parameters
        # server_learning_rate=1.0,
        # server_momentum=0.95,

        # FedOpt Parameters
        # eta = 1e-1,
        # eta_l = 1e-1,
        # beta_1 = 0.0,
        # beta_2 = 0.0,
        # tau = 1e-9,
    )

    # Start Flower server
    fl.server.start_server(
            "localhost:4687",
        config={"num_rounds": 10},
        strategy=strategy
    )


if __name__ == "__main__":
    main()
