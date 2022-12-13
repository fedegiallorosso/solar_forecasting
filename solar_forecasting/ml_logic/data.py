import numpy as np
import os
import pandas as pd
from transform_output_format import get_4D_output

def clean_data(X):
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """
    print("\n✅ data cleaned")

    return X

def get_chunk(index):

    file_path_X_train = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "raw_data", os.environ.get("FINE_NAME_X_TRAIN"))
    data_X_train = np.load(file_path_X_train, allow_pickle=True)

    file_path_Y_train = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "raw_data", os.environ.get("FINE_NAME_Y_TRAIN"))
    data_Y_train = pd.read_csv(file_path_Y_train)
    data_Y_train=get_4D_output(data_Y_train)

    month=[-1,123,248,403,571,757,937,1123,1306,1456,1600,1720,1844]

    # Extract features from data
    X_chunk=np.zeros((2,month[index]-month[index-1],4,81,81))
    Y_chunk=np.zeros((month[index]-month[index-1],4,51,51))

    df_date = pd.DataFrame ({'Datetime': data_X_train['datetime'][month[index-1]+1:month[index]+1]})
    X_chunk[0]=data_X_train['GHI'][month[index-1]+1:month[index]+1,:,:,:]
    X_chunk[1]=data_X_train['CLS'][month[index-1]+1:month[index]+1,:4,:,:]

    Y_chunk=data_Y_train[month[index-1]+1:month[index]+1,:,:,:]

    print(f"\n✅ Retrieving chunk n°{index}...")

    return X_chunk, df_date, Y_chunk
