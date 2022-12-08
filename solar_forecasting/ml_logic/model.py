
from colorama import Fore, Style

import time
start = time.perf_counter()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import Model, layers
from typing import Tuple

end = time.perf_counter()

def feature_construction(data, y):

    length=20

    X_train=np.zeros((length,4,51,51,2))
    for j in range (length):
        for i in range (4):
            for k in range (51):
                for h in range (51):
                    for z in range (2):
                        if z==0:
                            X_train[j,i,k,h,z]=data[0,j,i,k+15,h+15]
                        if z==1:
                            X_train[j,i,k,h,z]=data[1,j,i,k+15,h+15]

    X_val=np.zeros((data.shape[1]-length,4,51,51,2))
    for j in range (data.shape[1]-length):
        for i in range (4):
            for k in range (51):
                for h in range (51):
                    for z in range (2):
                        if z==0:
                            X_val[j,i,k,h,z]=data[0,j+length,i,k+15,h+15]
                        if z==1:
                            X_val[j,i,k,h,z]=data[0,j+length,i,k+15,h+15]

    y_train=np.zeros((length,4,51,51,2))
    for j in range (length):
        for i in range (4):
            for k in range (51):
                for h in range (51):
                    for z in range (2):
                        if z==0:
                            y_train[j,i,k,h,z]=y[j,i,k,h]
                        if z==1:
                            y_train[j,i,k,h,z]=y[j,i,k,h]

    y_val=np.zeros((data.shape[1]-length,4,51,51,2))
    for j in range (data.shape[1]-length):
        for i in range (4):
            for k in range (51):
                for h in range (51):
                    for z in range (2):
                        if z==0:
                            y_val[j,i,k,h,z]=y[j+length,i,k,h]
                        if z==1:
                            y_val[j,i,k,h,z]=y[j+length,i,k,h]

    return X_train, X_val, y_train, y_val

def initialize_model(X_train: np.ndarray) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    inp=layers.Input(shape=(None, *X_train.shape[2:]))

    x=layers.ConvLSTM2D(filters=64, kernel_size=(5,5),
                     padding='same',
                     return_sequences=True,
                     activation='relu',
                    )(inp)
    #x=layers.BatchNormalization()(x)
    #x=layers.Dropout(rate=0.1)(x)
    x=layers.Conv3D(filters=1, kernel_size=(3,3,3),
                activation="relu",
                padding="same")(x)
    model=keras.models.Model(inp,x)

    print("\n✅ model initialized")

    return model

def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile the Neural Network
    """
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=keras.losses.MeanAbsoluteError(),
                  metrics=keras.metrics.MeanAbsoluteError())

    print("\n✅ model compiled")
    return model

def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                batch_size=32,
                patience=5,
                epochs=100,
                ) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    #Define some callbacks to improve training
    early_stopping=keras.callbacks.EarlyStopping(monitor="val_loss",
                                                 patience=10,
                                                 restore_best_weights=True,
                                                 verbose=0)

    reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=patience)

    history= model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data= (X_val, y_val),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)

    print("\n✅ model trained ")

    return model, history


def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   chunk_id: int,
                   batch_size=64,
                   ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {chunk_id} chunk..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    mae = metrics["mean_absolute_error"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

    return metrics
