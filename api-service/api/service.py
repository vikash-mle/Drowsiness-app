from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import pandas as pd
import asyncio
import os
from fastapi import File
from tempfile import TemporaryDirectory
from api import model

# Initialize Tracker Service
#tracker_service = TrackerService()

# Setup FastAPI app
app = FastAPI(
    title="API Server",
    description="API Server",
    version="v1"
)

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
''''
@app.on_event("startup")
async def startup():
    # Startup tasks
    # Start the tracker service
    asyncio.create_task(tracker_service.track()) 


@app.on_event("shutdown")
async def shutdown():
    # Shutdown tasks 
    ... '''

# Routes
@app.get("/")
async def get_index():
    return {
        "message": "Hi there! I am API Service"
    }

@app.post("/predict")
async def predict(
        file: bytes = File(...)
):
    print("predict file:", len(file), type(file))

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.jpg")
        with open(image_path, "wb") as output:
            output.write(file)

        # Make prediction
        prediction_results = model.make_prediction(image_path)

    return prediction_results