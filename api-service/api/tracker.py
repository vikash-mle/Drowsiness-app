
import os
import asyncio
from glob import glob
import json
import pandas as pd

import tensorflow as tf
from google.cloud import storage


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/secrets/bucket-reader.json'  #set the env variable and address like this
gcp_project = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
bucket_name = "drowsiness-app-model"
local_experiments_path = "/persistent-folder/experiments"

# Setup experiments folder
if not os.path.exists(local_experiments_path):
    os.mkdir(local_experiments_path)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client(project=gcp_project)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)



def download_best_models():
    print("Download leaderboard models and artifacts")
    try:

            download_file = "cropping_face_with_aug.h5"
            download_blob(bucket_name, download_file,
                          os.path.join(local_experiments_path, download_file))


    except:
        print("No model found")


class TrackerService:
    def __init__(self):
        self.timestamp = 0

    async def track(self):
        while True:
            await asyncio.sleep(5)
            print("Tracking experiments...")
                # Download best model
            download_best_models()
