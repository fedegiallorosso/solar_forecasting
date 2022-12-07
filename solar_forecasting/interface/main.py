from solar_forecasting.ml_logic.data import (clean_data, get_chunk)
import numpy as np
import pandas as pd
from colorama import Fore, Style
from solar_forecasting.data_sources.get_data import download_file
import os

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
    chunk_id = 0  # monthly

    while (chunk_id<3):

        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        X_chunk = get_chunk(index=chunk_id)

        X_chunk_cleaned = clean_data(X_chunk)

        destination_name = f'../../raw_data/processed_data/{chunk_id+1}.npy'

        # save and append the chunk
        print (X_chunk_cleaned[0,:,0,0,0])

        with open(destination_name, 'wb') as f:
            np.save(f, X_chunk_cleaned)

        chunk_id += 1

    print("\n✅ data processed saved entirely:")

    return None

if __name__ == '__main__':
    download()
    preprocess()
