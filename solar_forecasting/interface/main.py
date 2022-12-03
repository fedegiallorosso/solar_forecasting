#from taxifare.ml_logic.data import (clean_data,
#                                    get_chunk,
#                                   save_chunk)

#from taxifare.ml_logic.params import (CHUNK_SIZE,
#                                      DATASET_SIZE,
#                                      VALIDATION_DATASET_SIZE)


from solar_forecasting.ml_logic.data import (clean_data, get_chunk)


#from taxifare.ml_logic.preprocessor import preprocess_features

import numpy as np
#import pandas as pd

from colorama import Fore, Style

def preprocess():
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    print("\n⭐️ use case: preprocess")

    # iterate on the dataset, by chunks
    chunk_id = 0  # from 0 to 11
    #row_count =
    #cleaned_row_count = 0
    source_name = '../../raw_data/baseline.npz' # f"{source_type}_{DATASET_SIZE}"
    destination_name = '../../raw_data/processed_data.npy'

    while (chunk_id<3):

        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        X_chunk = get_chunk(source_name=source_name,
                               index=chunk_id) #* CHUNK_SIZE,
                              # chunk_size=CHUNK_SIZE)

        # Break out of while loop if data is none
        #if X_chunk is None:
        #    print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
        #    break

    #    row_count += data_chunk.shape[0]

        X_chunk_cleaned = clean_data(X_chunk)

    #    cleaned_row_count += len(data_chunk_cleaned)

        # break out of while loop if cleaning removed all rows
    #    if len(data_chunk_cleaned) == 0:
    #        print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
    #        break

    #    X_chunk = data_chunk_cleaned.drop("fare_amount", axis=1)
    #    y_chunk = data_chunk_cleaned[["fare_amount"]]

    #    X_processed_chunk = preprocess_features(X_chunk)

    #    data_processed_chunk = pd.DataFrame(
    #        np.concatenate((X_processed_chunk, y_chunk), axis=1))

        # save and append the chunk
        print (X_chunk_cleaned[0,:,0,0,0])
        if chunk_id == 0:
            with open(destination_name, 'wb') as f:
                np.save(f, X_chunk_cleaned)
                f.close()
        else:
            with open(destination_name, 'ab') as f:
                np.save(f, X_chunk_cleaned)
                f.close()


    #    save_chunk(destination_name=destination_name,
    #               is_first=is_first,
    #               data=data_processed_chunk)

        chunk_id += 1

    #if row_count == 0:
    #    print("\n✅ no new data for the preprocessing 👌")
    #    return None

    print("\n✅ data processed saved entirely:")

    return None

if __name__ == '__main__':
    preprocess()
