from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import io
import fitz  # PyMuPDF

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    """Extracts text from PDF (kept here since it's lightweight)"""
    try:
        pdf_data = uploaded_file.file.read()
        if len(pdf_data) > 5_000_000:  # 5MB limit for Vercel
            return "Error: PDF too large (max 5MB)"
            
        pdf_stream = io.BytesIO(pdf_data)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@app.post("/match-jobs/")
async def match_jobs(file: UploadFile = File(...), query: str = Form(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Please upload a PDF file"}

    resume_text = extract_text_from_pdf(file)
    if not resume_text or "Error" in resume_text:
        return {"error": "Failed to extract text from resume"}

    try:
        # Call Render ML service
        ml_service_url = os.getenv("RENDER_ML_URL")
        response = requests.post(
            f"{ml_service_url}/predict",
            json={"resume_text": resume_text, "query": query},
            timeout=8  # Under Vercel's 10s limit
        )
        return response.json()
    except Exception as e:
        return {"error": f"Job matching failed: {str(e)}"}

@app.get("/get-jobs/")
async def get_jobs(query: str = Query(...)):
    """Directly calls JSearch API (no ML needed)"""
    api_key = os.getenv("JSEARCH_API_KEY")
    if not api_key:
        return {"error": "API key missing"}

    try:
        response = requests.get(
            "https://jsearch.p.rapidapi.com/search",
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            },
            params={"query": query, "num_pages": 1},
            timeout=8
        )
        data = response.json().get("data", [])
        return {
            "jobs": [{
                "title": job.get("job_title"),
                "company": job.get("employer_name"),
                "location": job.get("job_city"),
                "description": (job.get("job_description") or "")[:300] + "...",
                "apply_link": job.get("job_apply_link")
            } for job in data]
        }
    except Exception as e:
        return {"error": f"Failed to fetch jobs: {str(e)}"}