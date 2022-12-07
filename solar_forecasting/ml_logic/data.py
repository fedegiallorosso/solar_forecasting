import numpy as np
import os

def clean_data(X):
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """
    print("\n✅ data cleaned")

    return X

def get_chunk(index):

    file_path = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "raw_data", os.environ.get("FINE_NAME_X_TRAIN"))
    data = np.load(file_path, allow_pickle=True)

    month=[0,123,248,403, 571,757,937,1123,1306,1456,1600,1720,1844]

    # Extract features from data
    X_chunk={}
    X_chunk['DATE']=data['datetime'][month[index-1]+1:month[index]+1]
    X_chunk['GHI']=data['GHI'][month[index-1]+1:month[index]+1,:,:,:]
    X_chunk['CLS']=data['CLS'][month[index-1]+1:month[index]+1,:4,:,:]

    print(f"\n✅ Retrieving chunk n°{index}...")

    return X_chunk
