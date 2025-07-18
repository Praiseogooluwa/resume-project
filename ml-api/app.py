from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from matcher import get_top_matches
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(resume_text: str, query: str):
    try:
        return {
            "matches": get_top_matches(resume_text, query, top_k=3)
        }
    except Exception as e:
        return {"error": str(e)}