import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

def create_model(num_features):
    """
        Creates a deep autoencoder to process a tabular data with total features \
                represented by num_features.
    """
    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_features, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    return model

