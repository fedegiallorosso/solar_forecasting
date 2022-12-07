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

    # Extract features from data
    #GHI = data['GHI']
    CLS = data['CLS']
    SAA = data['SAA']
    SZA = data['SZA']
    #DATE = data['datetime']
    month=[0,20,50,100]
    index=index+1

    X_chunk=np.zeros((3,month[index]-month[index-1],8,81,81))
    print(X_chunk.shape)
    #X_chunk[0]=GHI[month[index-1]:month[index],:,:,:]
    X_chunk[0]=CLS[month[index-1]:month[index],:,:,:]
    X_chunk[1]=SAA[month[index-1]:month[index],:,:,:]
    X_chunk[2]=SZA[month[index-1]:month[index],:,:,:]

    return X_chunk
