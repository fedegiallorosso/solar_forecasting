from solar_forecasting.ml_logic.data import (clean_data, get_chunk)
import numpy as np
import pandas as pd
from colorama import Fore, Style
from solar_forecasting.data_sources.get_data import download_file
import matplotlib.pyplot as plt

def download():
    """
    Download the file from Google Cloud Storage using the function download_file
    """
    download_file()

    return None

def preprocess():
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """
    print("\n⭐️ Preprocess in progress ...")

    # iterate on the dataset, by chunks
    chunk_id = 1  # monthly

    while (chunk_id<13):

        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        X_chunk_feature, X_chunk_date, Y_chunk = get_chunk(index=chunk_id)

        #X_chunk_cleaned = clean_data(X_chunk)

        destination_name_feature = f'../../raw_data/processed_data/feature/{chunk_id}.npy'
        destination_name_date = f'../../raw_data/processed_data/date/{chunk_id}.csv'
        destination_name_Y = f'../../raw_data/processed_data/Y_train/{chunk_id}.npy'

        with open(destination_name_feature, 'wb') as f:
            np.save(f, X_chunk_feature)

        X_chunk_date.to_csv(destination_name_date, index=False)

        with open(destination_name_Y, 'wb') as f:
            np.save(f, Y_chunk)

        chunk_id += 1

    print("\n✅ data processed saved entirely:")

    return None

def train():
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """
    print("\n⭐️ Training in progress ...")

    from solar_forecasting.ml_logic.model import (feature_construction, initialize_model, compile_model, train_model)
    from solar_forecasting.ml_logic.registry import load_model, save_model

    model = None
    # model params
    learning_rate = 0.01
    batch_size = 64
    patience = 2
    epochs=100

    # iterate on the full dataset per chunks
    chunk_id = 1
    metrics_list = []

    while (chunk_id<10):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        data = np.load(f'../../raw_data/processed_data/feature/{chunk_id}.npy', allow_pickle=True)
        y = np.load(f'../../raw_data/processed_data/Y_train/{chunk_id}.npy')

        X_train, X_val, y_train, y_val = feature_construction (data=data, y=y)

        # initialize model
        if model is None:
            model = initialize_model(X_train)

        # (re)compile and train the model incrementally
        model = compile_model(model, learning_rate)
        model, history = train_model(model,
                                     X_train,
                                     y_train,
                                     X_val,
                                     y_val,
                                     batch_size=batch_size,
                                     patience=patience,
                                     epochs=epochs,
                                     )
        print(history.history.keys())

        metrics_chunk = np.min(history.history['mean_absolute_error'])
        metrics_list.append(metrics_chunk)
        print(f"chunk MAE: {round(metrics_chunk,2)}")

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'../../raw_data/MAE_Plot/{chunk_id}.png')

        chunk_id += 1

    # return the last value of the validation MAE
    mae = metrics_list[-1]

    print(f"\n✅ trained on {chunk_id-1} chucks with MAE: {round(mae, 2)}")

    params = dict(
        # model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        chunk=chunk_id,
        )

    #save model
    save_model(model=model, params=params, metrics=dict(mae=mae))

    return mae

def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Evaluate in progress...")

    from solar_forecasting.ml_logic.model import feature_construction, evaluate_model
    from solar_forecasting.ml_logic.registry import load_model, save_model

    chunk_id=10

    # load new data
    while (chunk_id<13):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        data = np.load(f'../../raw_data/processed_data/feature/{chunk_id}.npy', allow_pickle=True)
        y = np.load(f'../../raw_data/processed_data/Y_train/{chunk_id}.npy')

        X_train, X_val, y_train, y_val = feature_construction(data=data, y=y)

        if X_train is None:
            print("\n✅ no data to evaluate")
            return None

        model = load_model()

        metrics_dict = evaluate_model(model=model, X=X_train, y=y_train, chunk_id=chunk_id )
        mae = metrics_dict["mean_absolute_error"]

        # save evaluation
        params = dict(        # package behavior
            context="evaluate",
            # data source
            chunk_id=chunk_id)

        chunk_id += 1

    save_model(params=params, metrics=dict(mae=mae))

    return mae

if __name__ == '__main__':
    download()
    #preprocess()
    #train()
    #evaluate()
    #pred()
