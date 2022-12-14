from solar_forecasting.ml_logic.data import (clean_data, get_chunk)
import numpy as np
import pandas as pd
from colorama import Fore, Style
from solar_forecasting.data_sources.get_data import download_file
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import ipdb



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
    from solar_forecasting.ml_logic.registry import save_model
    from solar_forecasting.data_sources.preprocessing import number_of_observations

    model = None
    # model params
    learning_rate = 0.001
    batch_size = 8
    patience = 2
    epochs=4 # should be 5

    # iterate on the full dataset per chunks
    chunk_id = 1
    loss_history = []
    val_loss_history = []
    metrics_list = []

    while (chunk_id<2):  #Training + Validation until the end of July

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        #Loading the files
        feature = np.load(f'../../raw_data/processed_data/feature/{chunk_id}.npy', allow_pickle=True)
        y = np.load(f'../../raw_data/processed_data/Y_train/{chunk_id}.npy')
        DATE = pd.read_csv(f'../../raw_data/processed_data/date/{chunk_id}.csv')
        #Converting column 'Datetime' to datetime format
        DATE['Datetime']=pd.to_datetime(DATE['Datetime'])
        #Retrieving the date
        number_of_observation=0

        while number_of_observation < len(DATE):

            print (number_of_observation)
            if number_of_observation < len(DATE) :
                number_of_observation_train = number_of_observations((DATE['Datetime'][number_of_observation]).date())
            else:
                break
            print ((DATE['Datetime'][number_of_observation]).date())

            print (number_of_observation_train)
            if number_of_observation_train+number_of_observation < len(DATE) :
                number_of_observation_val = number_of_observations((DATE['Datetime'][number_of_observation_train+number_of_observation]).date())
            else:
                break

            print ((DATE['Datetime'][number_of_observation_train+number_of_observation]).date())
            print (number_of_observation_val)

            X_train, X_val, y_train, y_val = feature_construction (feature, y, DATE,
                                                                   number_of_observation_train,
                                                                   number_of_observation_val)
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

            metrics_chunk = np.min(history.history['mean_absolute_error'])
            metrics_list.append(metrics_chunk)
            loss_history.append(history.history['loss'])
            val_loss_history.append(history.history['val_loss'])

            #output = pd.DataFrame ({"Metrics": metrics_list, "Loss":loss_history, "Val_Loss": val_loss_history })
            #output.to_csv('../../raw_data/Results.csv')

            number_of_observation += number_of_observation_train + number_of_observation_val

        plt.subplot()
        # Graph History for Loss
        plt.plot(loss_history)
        plt.plot(val_loss_history)
        plt.title('Model loss')
        plt.ylabel('Loss [MAE]')
        plt.xlabel('Day')
        plt.xticks([1,30])
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'../../raw_data/Model_loss.png')

        # return the last value of the validation MAE
        mae = metrics_list[-1]

        print(f"\n✅ trained on {chunk_id-1} chucks chuck with MAE: {round(mae, 2)}")

        params = dict(
        # model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        )

        #save model
        save_model(model=model, params=params, metrics=dict(mae=mae))

        chunk_id += 1

    return None


def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ Evaluate in progress...")

    from solar_forecasting.ml_logic.model import feature_construction, evaluate_model
    from solar_forecasting.ml_logic.registry import load_model, save_model
    from solar_forecasting.data_sources.preprocessing import number_of_observations

    chunk_id=8 #Evaluation on August, September, October

    # load new data
    while (chunk_id<11):

        print(Fore.BLUE + f"\nLoading and evaluating the model on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        #Loading the files
        feature = np.load(f'../../raw_data/processed_data/feature/{chunk_id}.npy', allow_pickle=True)
        y = np.load(f'../../raw_data/processed_data/Y_train/{chunk_id}.npy')
        DATE = pd.read_csv(f'../../raw_data/processed_data/date/{chunk_id}.csv')
        #Converting column 'Datetime' to datetime format
        DATE['Datetime']=pd.to_datetime(DATE['Datetime'])
        #Retrieving the date
        number_of_observation=0
        model = None

        #Defining the metrics
        mae=[]

        while number_of_observation < len(DATE):

            print (number_of_observation)
            if number_of_observation < len(DATE) :
                number_of_observation_train = number_of_observations((DATE['Datetime'][number_of_observation]).date())
            else:
                break
            print ((DATE['Datetime'][number_of_observation]).date())

            print (number_of_observation_train)
            if number_of_observation_train+number_of_observation < len(DATE) :
                number_of_observation_val = number_of_observations((DATE['Datetime'][number_of_observation_train+number_of_observation]).date())
            else:
                break

            print ((DATE['Datetime'][number_of_observation_train+number_of_observation]).date())
            print (number_of_observation_val)

            X_train, X_val, y_train, y_val = feature_construction (feature, y, DATE,
                                                                   number_of_observation_train,
                                                                   number_of_observation_val)
            if X_train is None:
                print("\n✅ no data to evaluate")
                return None

            if model is None:
                model = load_model()

            metrics_dict = evaluate_model(model=model, X=X_train, y=y_train, chunk_id=chunk_id)
            mae.append(metrics_dict["mean_absolute_error"])

            # save evaluation
            #params = dict(context="evaluate",chunk_id=chunk_id)

            number_of_observation += number_of_observation_train + number_of_observation_val

            print(f"Evaluation on chunk {chunk_id} date {(DATE['Datetime'][number_of_observation]).date()} done! MAE: {round(mae[-1],2)}")

        chunk_id += 1

    #save_model(params=params, metrics=dict(mae=mae))

    return None

def pred():
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Prediction in progress...")

    from solar_forecasting.ml_logic.registry import load_model
    from solar_forecasting.ml_logic.model import feature_construction_prediction
    from solar_forecasting.data_sources.preprocessing import number_of_observations

    chunk_id=11 #EPrediction on November and December

    print(Fore.BLUE + f"\nLoading and making predictions on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

    feature = np.load(f'../../raw_data/processed_data/feature/{chunk_id}.npy', allow_pickle=True)
    DATE = pd.read_csv(f'../../raw_data/processed_data/date/{chunk_id}.csv')

    #Converting column 'Datetime' to datetime format
    DATE['Datetime']=pd.to_datetime(DATE['Datetime'])
    #Retrieving the date
    number_of_observation=0

    number_of_observation_train = number_of_observations((DATE['Datetime'][number_of_observation]).date())
    actual_day= (DATE['Datetime'][number_of_observation]).date()

    GHI=feature[0]
    print (GHI.shape)
    GHI=GHI[:1,:,15:66,15:66]
    print (GHI.shape)
    X_train=np.expand_dims(GHI, axis=-1)
    model = load_model()

    while number_of_observation_train <=4:

        print (X_train.shape)

        y_pred = model.predict(X_train)
        print (y_pred.shape)
        y_pred=np.squeeze(y_pred, axis=-1)
        print (y_pred.shape)
        y_pred=y_pred[:,3:4,:,:]
        print (y_pred.shape)

        #pred_image = y_pred[0,-1,:,:,:]

        #X_train=X_train[:,:-3,:,:,:]
        #print (X_train.shape)

        #X_train=np.append(X_train, pred_image).reshape(4, 4, 51, 51, 1)
        #print (X_train.shape)

        number_of_observation_train+=1

    print("\n✅ Prediction done! ")

    #print (y_pred.shape)







    #chunk_id = 12
    #prediction = np.load(f'../../raw_data/Prediction{chunk_id}.npy', allow_pickle=True)
    orig_data = np.load(f'../../raw_data/X_train.npz', allow_pickle=True)
    #DATE = orig_data['datetime']

    #GHI_PRED = np.zeros((prediction.shape[0],4,51,51))

    #for i in range (GHI_PRED.shape[0]):
    #    for j in range (GHI_PRED.shape[1]):
    #        for z in range (GHI_PRED.shape[2]):
    #            for t in range (GHI_PRED.shape[3]):
    #                GHI_PRED[i,j,z,t]=prediction[i,j,z,t,0]

    #print (GHI_PRED.shape)

    from solar_forecasting.data_sources.preprocessing import number_of_observations, date_index

    #actual_day = input("Write the date ")
    observations = number_of_observations(actual_day)

    first_index = date_index(actual_day)[0]
    #time = DATE[first_index]
    #time = time - datetime.timedelta(minutes=60)
    delta=datetime.timedelta(minutes=15)

    vmin=(orig_data['GHI'][first_index:first_index+observations,:,:,:]).min()
    vmax=(orig_data['GHI'][first_index:first_index+observations,:,:,:]).max()
    print (observations)
    print (first_index)
    print (first_index-1721)

    plt.figure (figsize = (150,100))
    plt.title(f"Displaying the full day on {actual_day}", fontsize=100)

    for j in range(observations):
        for i in range(8):
            time=time+delta
            ax = plt.subplot(observations, 8, j * 8 + i + 1)
            plt.title(time.time(), fontsize=100)
            if i < 4:
                pic = ax.imshow(orig_data['GHI'][first_index + j, i, :, :], cmap='jet', vmin=vmin, vmax=vmax)
            #else:
                #pic = ax.imshow(GHI_PRED[first_index-1721+j, i-4, :, :], cmap='jet', vmin=vmin, vmax=vmax)

    plt.colorbar(pic)
    plt.savefig(f'../../raw_data/Pred.png')

    return None

if __name__ == '__main__':
    download()
    preprocess()
    train()
    #evaluate()
    #pred()
    #plotting_prediction()
