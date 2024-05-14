import logging
from typing import List

import numpy as np
from bson import ObjectId
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import logging as cloud_logging
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from scipy.stats import ks_2samp

from pipeline import model_inference_step

app = FastAPI()

# Set up Google Cloud Logging
cloud_logging_client = cloud_logging.Client()
cloud_logging_client.setup_logging()

MONGO_DETAILS = "mongodb+srv://adnanekesraoui:adnane2000@sentiments.n2yephg.mongodb.net/"
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client.sentiments
collection = db.sentiment_data

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str


def check_data_drift(reference_data, new_data, threshold=0.05):
    statistic, p_value = ks_2samp(reference_data, new_data)
    logging.info(f"K-S Test statistic: {statistic}, p-value: {p_value}")
    if p_value < threshold:
        logging.warning("Data drift detected")
    else:
        logging.info("No data drift detected")

@app.post("/predict")
async def predict_sentiment_file(file: UploadFile = File(...)):
    contents = await file.read()
    texts = contents.decode('utf-8').splitlines()
    results = []

    reference_sentiment_distribution = np.random.normal(0.5, 0.1, 1000)  

    current_sentiments = []

    for text in texts:
        try:
            prediction = model_inference_step([text])  
            sentiment_document = {
                "text": text,
                "sentiment": prediction[0]
            }
            insert_result = await collection.insert_one(sentiment_document)  
            sentiment_document['_id'] = str(insert_result.inserted_id)  

            sentiment_value = 1 if prediction[0] == "positive" else -1 if prediction[0] == "negative" else 0
            current_sentiments.append(sentiment_value)

            logging.info(f"Predicted sentiment: {prediction[0]} for text: {text}")
        except Exception as e:
            logging.error(f"Error predicting sentiment for text: {text} - {str(e)}")

    check_data_drift(reference_sentiment_distribution, current_sentiments)

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
