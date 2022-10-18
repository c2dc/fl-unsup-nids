import flwr as fl
import tensorflow as tf
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
from utils import model, load_data
from efc import EnergyBasedFlowClassifier

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def eval_learning(y_test, preds):
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    missrate = fn / (fn + tp)
    fallout = fp / (fp + tn)
    auc = roc_auc_score(y_test, preds)
    return acc, rec, prec, f1, mcc, missrate, fallout, auc


def distance_calc(losses, benign, attack):
    # for each sample loss, calculate the minimun distance and set a label for test purpose
    result = np.zeros(len(losses))
    for i, loss in enumerate(losses):
        if abs(loss - benign) > abs(loss - attack):
            result[i] = 1
        else:
            result[i] = 0

    return result


def calculate_reconstruction_loss(x, x_hat):
    losses = np.mean(abs(x - x_hat), axis=1)  # MAE
    return losses


class FlwrClient(fl.client.NumPyClient):
    # Implement flower client extending class NumPyClient
    # This class serialize and de-serialize the weights into numpy ndarray

    def __init__(self, cid, with_efc=False, full=False):
        self.cid = cid
        self.pre_cid = cid[cid.rfind("/") + 1:cid.find(".csv")]
        self.x_train, self.y_train, self.x_test, self.y_test = load_data.load_data(self.cid, full=full)

        if with_efc:
            self.model = model.create_model(self.x_train.shape[1] + 1)  # consider one additional feature (EFC energy)
            efc = EnergyBasedFlowClassifier(cutoff_quantile=0.95)
            efc.fit(self.x_train, self.y_train)
            _, y_energies_train = efc.predict(self.x_train, return_energies=True)
            _, y_energies_test = efc.predict(self.x_test, return_energies=True)

            self.x_train = np.append(self.x_train, y_energies_train.reshape(y_energies_train.shape[0], 1), axis=1)
            self.x_test = np.append(self.x_test, y_energies_test.reshape(y_energies_test.shape[0], 1), axis=1)
        else:
            self.model = model.create_model(self.x_train.shape[1])

        self.loss = 0
        self.threshold_benign = 0  # this threshold is calculated o % benign samples from train data (during evaluate)
        self.threshold_attack = 0  # this threshold is calculated on attack samples from train data (during evaluate)

        train_data = self.x_train[self.y_train == 0]  # only benign samples
        idx = int(train_data.shape[0] * 0.9)
        self.val_data = train_data[idx:]  # holdout validation set for threshold calculation
        self.train_data = train_data[:idx]  # reduced self.x_train (only benign wo val_data)
        self.attack_data = self.x_train[
            self.y_train == 1]  # the attack data from train set (used only for threshold estimation)

    def get_parameters(self):
        # return local model parameters
        return self.model.get_weights()  # get_weights from keras returns the weights as ndarray

    def set_parameters(self, parameters):
        # Server sets model parameters from a list of NumPy ndarrays (Optional)
        self.model.set_weights(parameters)  # set_weights on local model (similar to get_weights)

    def fit(self, parameters, config):
        # receive parameters from the server and use them to train on local data (locally)
        self.set_parameters(parameters)

        # https://keras.io/api/models/model_training_apis/#fit-method
        history = self.model.fit(
            self.train_data,
            self.train_data,
            batch_size=128,  # config["batch_size"],
            shuffle=True,
            epochs=10  # config["num_epochs"] # single epoch on local data
        )

        # return the refined model parameters with get_weights, the length of local data,
        #        and custom metrics can be provided through dict
        # len(x_train) is a useful information for FL, analogous to weights of contribution of each client
        self.loss = history.history["loss"][-1]

        return self.get_parameters(), len(self.train_data), {"loss": history.history["loss"][-1], }

    def evaluate(self, parameters, config):
        # evaluates the model on local data (locally)
        self.set_parameters(parameters)

        # eval new model on holdout validation set
        val_inference = self.model.predict(self.val_data)
        attack_inference = self.model.predict(self.attack_data)
        val_losses = calculate_reconstruction_loss(self.val_data, val_inference)
        attack_losses = calculate_reconstruction_loss(self.attack_data, attack_inference)

        #########################
        # Threshold Calculation #
        #########################
        self.threshold_benign = np.mean(val_losses)
        # self.threshold_benign = np.quantile(val_losses, 0.95)
        self.threshold_attack = np.mean(attack_losses)

        print("\n>> {} Mean Validation Loss (Benign): {} | (Attack): {}".format(self.pre_cid, self.threshold_benign,
                                                                                self.threshold_attack))

        # Test Set Evaluation
        inference = self.model.predict(self.x_test)
        losses = calculate_reconstruction_loss(self.x_test, inference)

        ######################
        # Threshold Criteria #
        ######################
        test_eval = distance_calc(losses, self.threshold_benign, self.threshold_attack)
        # test_eval = losses > self.threshold_benign

        acc, rec, prec, f1, mcc, missrate, fallout, auc = eval_learning(self.y_test, test_eval)

        compare = pd.DataFrame([yreal == yhat for yreal, yhat in zip(self.y_test, test_eval)], columns=["Predict"])
        compare.to_csv("./extra/" + self.pre_cid + "_predicts.csv", index=False)

        output_dict = {"acc": acc, "rec": rec, "prec": prec, "f1": f1, "mcc": mcc, "missrate": missrate,
                "fallout": fallout, "auc": auc}

        print("\n>> Evaluate {} | Threshold {} | Metrics {} | Mean_Attack_Loss {} | Mean_Benign_Loss {}".format(
            self.pre_cid, self.threshold_benign, output_dict, np.mean(losses[self.y_test == 1]),
            np.mean(losses[self.y_test == 0])))

        return float(self.loss), len(self.x_test), output_dict


def main():
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--silo", type=str, required=True)
    parser.add_argument("--with_efc", required=False, action="store_true")
    parser.add_argument("--full", required=False, action="store_true")
    args = parser.parse_args()

    # Start Flower client		
    client = FlwrClient(args.silo, with_efc=args.with_efc, full=args.full)

    fl.client.start_numpy_client(server_address="[::]:4687", client=client)


if __name__ == "__main__":
    main()
