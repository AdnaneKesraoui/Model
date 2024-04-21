from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import model_inference_step

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(text_request: TextRequest):
    text = text_request.text
    predictions = model_inference_step([text])  # Only passing text to inference step
    return {"sentiment": predictions[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
