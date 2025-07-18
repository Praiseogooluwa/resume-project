from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import io
import fitz  # PyMuPDF
from typing import Dict, List

app = FastAPI(title="Resume Matcher API", 
             description="Matches resumes to jobs using AI",
             version="1.0")

# Enhanced CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    max_age=600
)

def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    """Safely extracts text from PDF with size validation"""
    try:
        # Read first 5MB to check if PDF
        pdf_data = uploaded_file.file.read(5_000_000)
        if len(pdf_data) == 5_000_000:
            if uploaded_file.file.read(1):  # Check if there's more data
                return "Error: PDF exceeds 5MB limit"
        
        pdf_stream = io.BytesIO(pdf_data)
        try:
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            return "".join(page.get_text() for page in doc)
        except fitz.FileDataError:
            return "Error: Invalid PDF file"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@app.post("/match-jobs/", 
          response_model=Dict[str, List[Dict]], 
          summary="Match resume to jobs")
async def match_jobs(file: UploadFile = File(...), 
                    query: str = Form(...)) -> Dict:
    """
    Processes PDF resume and returns matching jobs using AI.
    - **file**: PDF resume (max 5MB)
    - **query**: Job search keywords (e.g. "software engineer")
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are accepted")

    resume_text = extract_text_from_pdf(file)
    if not resume_text or "Error" in resume_text:
        raise HTTPException(400, detail=resume_text.replace("Error: ", ""))

    try:
        # Call Render ML service with retries
        ml_service_url = os.getenv("RENDER_ML_URL")  # Configure in Vercel
        if not ml_service_url:
            raise HTTPException(500, detail="ML service URL not configured")

        response = requests.post(
            f"{ml_service_url}/predict",
            json={"resume_text": resume_text, "query": query},
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, detail=f"ML service error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")

@app.get("/get-jobs/", 
         response_model=Dict[str, List[Dict]], 
         summary="Fetch live job listings")
async def get_jobs(query: str = Query(..., min_length=2)) -> Dict:
    """
    Fetches live job listings from JSearch API.
    - **query**: Job search keywords (e.g. "python developer")
    """
    api_key = os.getenv("JSEARCH_API_KEY")
    if not api_key:
        raise HTTPException(500, detail="JSearch API key not configured")

    try:
        response = requests.get(
            "https://jsearch.p.rapidapi.com/search",
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            },
            params={"query": query, "num_pages": "1"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        
        return {
            "jobs": [{
                "title": job.get("job_title", "No title"),
                "company": job.get("employer_name", "Unknown"),
                "location": f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(", "),
                "description": (job.get("job_description") or "")[:300] + "...",
                "apply_link": job.get("job_apply_link") or "#"
            } for job in data]
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, detail=f"JSearch API error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")
