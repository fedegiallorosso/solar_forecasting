
from colorama import Fore, Style

import time
start = time.perf_counter()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import Model, layers
from typing import Tuple
import math
import datetime

end = time.perf_counter()

def feature_construction(feature, y, DATE, number_of_observation_per_day_train, number_of_observation_per_day_val ):

    #Defining the features
    GHI=feature[0,:number_of_observation_per_day_train+number_of_observation_per_day_val,:,15:66,15:66]
    y=y[:number_of_observation_per_day_train+number_of_observation_per_day_val,:,:,:]
    #CLS=feature[1,:,:,:,:]

    #Preparing the dataframe used for cyclical encoding for feature "date"
    #DATE['Datetime']=pd.to_datetime(DATE['Datetime'])
    #df = pd.DataFrame({"time":pd.date_range(start='05:30', end='17:30', freq='15min')[:-1]})
    #df["x"]=np.linspace(0, 12 * 4 - 1, 12 * 4, dtype=int)
    # We normalize x values to match with the 0-2π cycle
    #df["x_norm"] = 2 * math.pi * df["x"] / df["x"].max()
    #df["cos_x"] = np.cos(df["x_norm"])
    #df["sin_x"] = np.sin(df["x_norm"])
    #delta=datetime.timedelta(minutes=15)
    #hour=datetime.timedelta(minutes=45)

    #Defining the size of each dimension of the model input
    number_of_timestamp_per_obs=4
    image_width=51
    image_heigt=51
    number_of_features=1

    #Instantiate the model input
    X_train=np.zeros((8*number_of_observation_per_day_train,number_of_timestamp_per_obs,image_width,image_heigt,number_of_features))
    y_train=np.zeros((8*number_of_observation_per_day_train,number_of_timestamp_per_obs,image_width,image_heigt,number_of_features))
    X_val=np.zeros((8*number_of_observation_per_day_val,number_of_timestamp_per_obs,image_width,image_heigt,number_of_features))
    y_val=np.zeros((8*number_of_observation_per_day_val,number_of_timestamp_per_obs,image_width,image_heigt,number_of_features))

    #Building the complete train day
    complete_train_day=[]
    for i in range (number_of_observation_per_day_train):
        for j in range (4):
            complete_train_day.append(GHI[i,j,:,:])
        for z in range (4):
            complete_train_day.append(y[i,z,:,:])

    #Building the complete val day
    complete_val_day=[]
    for i in range (number_of_observation_per_day_val):
        for j in range (4):
            complete_val_day.append(GHI[i+number_of_observation_per_day_train,j,:,:])
        for z in range (4):
            complete_val_day.append(y[i+number_of_observation_per_day_train,z,:,:])

    #Building X train and y train
    for j in range (1,8*number_of_observation_per_day_train-4):
        X_train[j]=np.expand_dims(complete_train_day[j:j+4], axis=-1)
    for j in range (2,8*number_of_observation_per_day_train-3):
        y_train[j]=np.expand_dims(complete_train_day[j:j+4], axis=-1)

    #Building X val and y val
    for j in range (1,8*number_of_observation_per_day_val-4):
        X_train[j]=np.expand_dims(complete_val_day[j:j+4], axis=-1)
    for j in range (2,8*number_of_observation_per_day_val-3):
        y_train[j]=np.expand_dims(complete_val_day[j:j+4], axis=-1)

    return X_train, X_val, y_train, y_val

def feature_construction_prediction(feature, DATE, number_of_observation_per_day_train):

    #Defining the features
    GHI=feature[0,:number_of_observation_per_day_train,:,15:66,15:66]

    #CLS=feature[1,:,:,:,:]

    #Preparing the dataframe used for cyclical encoding for feature "date"
    #DATE['Datetime']=pd.to_datetime(DATE['Datetime'])
    #df = pd.DataFrame({"time":pd.date_range(start='05:30', end='17:30', freq='15min')[:-1]})
    #df["x"]=np.linspace(0, 12 * 4 - 1, 12 * 4, dtype=int)
    # We normalize x values to match with the 0-2π cycle
    #df["x_norm"] = 2 * math.pi * df["x"] / df["x"].max()
    #df["cos_x"] = np.cos(df["x_norm"])
    #df["sin_x"] = np.sin(df["x_norm"])
    #delta=datetime.timedelta(minutes=15)
    #hour=datetime.timedelta(minutes=45)

    #Defining the size of each dimension of the model input
    #number_of_timestamp_per_obs=4
    #image_width=51
    image_heigt=51
    number_of_features=1

    #Instantiate the model input
    X_train=np.expand_dims(GHI, axis=-1)

    return X_train

def initialize_model(X_train: np.ndarray) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    inp=layers.Input(shape=(None, *X_train.shape[2:]))

    x= layers.ConvLSTM2D(filters=64, kernel_size=(5,5),
                        padding='same',
                        return_sequences=True,
                        activation='relu',
                        )(inp)
    #x=layers.BatchNormalization()(x)
    #x= layers.ConvLSTM2D(filters=64, kernel_size=(3,3),
    #                    padding='same',
    #                    return_sequences=True,
    #                    activation='relu',
    #                    )(inp)
    x=layers.Conv3D(filters=1, kernel_size=(3,3,3),
                    activation="relu",
                    padding="same")(x)
    # we will build the complete model and compile it
    model=keras.models.Model(inp,x)

    print("\n✅ model initialized")

    return model

def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile the Neural Network
    """
    optimizer=keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=keras.losses.MeanAbsoluteError(),
                  metrics=keras.metrics.MeanAbsoluteError())

    print("\n✅ model compiled")
    return model

def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                batch_size,
                patience,
                epochs,
                ) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    #Define some callbacks to improve training
    early_stopping=keras.callbacks.EarlyStopping(monitor="val_loss",
                                                 patience=patience,
                                                 restore_best_weights=False,
                                                )
    history= model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data= (X_val, y_val),
                        callbacks=[early_stopping],
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
