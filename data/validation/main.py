from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import model_inference_step

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

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
