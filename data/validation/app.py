from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from pipeline import model_inference_step

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

@app.post("/predict")
async def predict_sentiment(text_request: TextRequest):
    text = text_request.text
    predictions = model_inference_step([text])  # assume this returns the sentiment prediction

    sentiment_document = {
        "text": text,
        "sentiment": predictions[0]
    }
    
    await collection.insert_one(sentiment_document)  # insert the document into MongoDB
    return {"sentiment": predictions[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
