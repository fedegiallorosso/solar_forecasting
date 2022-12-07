from google.cloud import storage
import os
#from solar_forecasting.ml_logic.params import BUCKET_NAME, BLOB_NAME

def download_file(bucket_name = os.environ.get("BUCKET_NAME"),
                    blob_name_X_train = os.environ.get("BLOB_NAME_X_TRAIN"),
                    blob_name_Y_train = os.environ.get("BLOB_NAME_Y_TRAIN"),
                    download_to_disk = True,
                    destination_file_name_X_train = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "raw_data", os.environ.get("FINE_NAME_X_TRAIN")),
                    destination_file_name_Y_train = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "raw_data", os.environ.get("FINE_NAME_Y_TRAIN")),
                    ):

    """Download a file from Google Cloud Storage.
    If download_to_disk = True then it will save to your local disk.
    If download_to_disk = False then it will save to memory.
    """
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob_X_train = bucket.blob(blob_name_X_train)
    blob_Y_train = bucket.blob(blob_name_Y_train)

    if download_to_disk == True:

        print("Download to disk in progress ...")

        blob_X_train.download_to_filename(destination_file_name_X_train)

        print("Downloaded storage object {} \n from bucket {} \n to local file {}.".format(
            blob_name_X_train, bucket_name, destination_file_name_X_train))

        blob_Y_train.download_to_filename(destination_file_name_Y_train)

        print("Downloaded storage object {} \n from bucket {} \n to local file {}.".format(
            blob_name_Y_train, bucket_name, destination_file_name_Y_train))

    if download_to_disk == False:

        print("Download to memory in progress ...")

        contents = blob_X_train.download_as_string()

        print("Downloaded storage object {} from bucket {} as the following string: {}.".format(
            blob_name_X_train, bucket_name, contents
        )
    )

    return None
