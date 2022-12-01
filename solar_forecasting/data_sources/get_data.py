from google.cloud import storage
import os

#from solar_forecasting.ml_logic.params import BUCKET_NAME, BLOB_NAME

def download_file(bucket_name = os.environ.get("BUCKET_NAME"),
                    blob_name = os.environ.get("BLOB_NAME"),
                    download_to_disk = False,
                    destination_file_name = '../../raw_data/datafede2.npz'):

    """Download a file from Google Cloud Storage.
    If download_to_disk = False then it will save to memory.
    If download_to_disk = True then it will save to your local disk.
    """
    # print (bucket_name)
    # print (blob_name)

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(blob_name)

    if download_to_disk == True:

        blob.download_to_filename(destination_file_name, end=6000000)
        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, bucket_name, destination_file_name
        )
    )

    if download_to_disk == False:

        contents = blob.download_as_string()

        print("Downloaded storage object {} from bucket {} as the following string: {}.".format(
            blob_name, bucket_name, contents
        )
    )

    return "Download complete!"

if __name__ == "__main__":
    download_file(download_to_disk = True)
    print("Download complete!")
