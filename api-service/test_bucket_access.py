import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/secrets/bucket-reader.json'  #set the env variable and address like this
gcp_project = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
bucket_name = "drowsiness-app-model"
persistent_folder = "/persistent"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client(project=gcp_project)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


print(gcp_project)

# Test access
download_file = "test-bucket-access.txt"
download_blob(bucket_name, download_file,
              os.path.join(persistent_folder, download_file))