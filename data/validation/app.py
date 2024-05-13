from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from pipeline import model_inference_step, preprocess_step

app = FastAPI()

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

from typing import List

from bson import ObjectId  # Import this at the top of your file
from fastapi import FastAPI, File, UploadFile


@app.post("/predict_file")
async def predict_sentiment_file(file: UploadFile = File(...)):
    contents = await file.read()
    texts = contents.decode('utf-8').splitlines()
    results = []

    for text in texts:
        preprocessed_text = preprocess_step([text])  # Ensure text is passed as a list
        prediction = model_inference_step(preprocessed_text)  # Assume this function can handle a list of texts
        sentiment_document = {
            "text": text,
            "sentiment": prediction[0]
        }
        insert_result = await collection.insert_one(sentiment_document)  # Insert the document into MongoDB
        sentiment_document['_id'] = str(insert_result.inserted_id)  # Convert ObjectId to string
        results.append(sentiment_document)

    return results




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
