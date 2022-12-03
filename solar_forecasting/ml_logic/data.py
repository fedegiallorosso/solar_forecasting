#from taxifare.ml_logic.params import (COLUMN_NAMES_RAW,
#                                           DTYPES_RAW_OPTIMIZED,
#                                            DTYPES_RAW_OPTIMIZED_HEADLESS,
#                                            DTYPES_PROCESSED_OPTIMIZED
#                                            )

#from taxifare.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)

#from taxifare.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

#import os
#import pandas as pd
import numpy as np
import os


def clean_data(X):
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """

    # remove useless/redundant columns
    #df = df.drop(columns=['key'])

    # remove buggy transactions
    #df = df.drop_duplicates()  # TODO: handle in the data source if the data is consumed by chunks
    #df = df.dropna(how='any', axis=0)
    #df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) |
    #        (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    #df = df[df.passenger_count > 0]
    #df = df[df.fare_amount > 0]

    # remove irrelevant/non-representative transactions (rows) for a training set
    #df = df[df.fare_amount < 400]
    #df = df[df.passenger_count < 8]
    #df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
    #df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
    #df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
    #df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

    print("\n✅ data cleaned")

    return X

def get_chunk(source_name,
              index):
       #       chunk_size: int = None,
       #       verbose=False) -> pd.DataFrame:
#    """
#    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
#    Always assumes `source_name` (CSV or Big Query table) have headers,
#    and do not consider them as part of the data `index` count.
#    """

#    if "processed" in source_name:
#        columns = None
#        dtypes = DTYPES_PROCESSED_OPTIMIZED
#    else:
#        columns = COLUMN_NAMES_RAW
#        if os.environ.get("DATA_SOURCE") == "big query":
#            dtypes = DTYPES_RAW_OPTIMIZED
#        else:
#            dtypes = DTYPES_RAW_OPTIMIZED_HEADLESS

    # Load data from raw_data
    #print (os.getcwd())
    data = np.load(source_name, allow_pickle=True)

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





#    if os.environ.get("DATA_SOURCE") == "big query":

#        chunk_df = get_bq_chunk(table=source_name,
#                                index=index,
#                                chunk_size=chunk_size,
#                                dtypes=dtypes,
#                                verbose=verbose)



#        return chunk_df

#    chunk_df = get_pandas_chunk(path=source_name,
#                                index=index,
#                                chunk_size=chunk_size,
#                                dtypes=dtypes,
#                                columns=columns,
#                                verbose=verbose)

#    return chunk_df

"""
def save_chunk(destination_name: str,
               is_first: bool,
               data: pd.DataFrame) -> None:

    save chunk


    if os.environ.get("DATA_SOURCE") == "big query":

        save_bq_chunk(table=destination_name,
                      data=data,
                      is_first=is_first)

        return

    save_local_chunk(path=destination_name,
                     data=data,
                     is_first=is_first)
"""
