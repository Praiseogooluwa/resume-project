from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import requests
import os
import io
import fitz  # PyMuPDF
from typing import Dict, List

# Create FastAPI app
app = FastAPI(
    title="Resume Matcher API",
    description="AI-powered job matching system",
    version="2.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Resume Matcher API is running",
        "version": "2.1",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "match_jobs": "/match-jobs/",
            "get_jobs": "/get-jobs/"
        }
    }

def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    try:
        uploaded_file.file.seek(0)
        pdf_data = uploaded_file.file.read()
        
        if len(pdf_data) > 5_000_000:
            return "Error: PDF exceeds 5MB limit"

        pdf_stream = io.BytesIO(pdf_data)
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@app.post("/match-jobs/")
async def match_jobs(
    file: UploadFile = File(..., description="PDF resume file"),
    query: str = Form(..., description="Job search query")
):
    """Match resume with jobs using ML service"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    resume_text = extract_text_from_pdf(file)
    if "Error" in resume_text:
        raise HTTPException(status_code=400, detail=resume_text.replace("Error: ", ""))

    try:
        ml_url = os.getenv("RENDER_ML_URL")
        if not ml_url:
            raise HTTPException(status_code=500, detail="ML service URL not configured")

        response = requests.post(
            f"{ml_url.rstrip('/')}/predict",
            json={
                "resume_text": resume_text,
                "query": query.strip()
            },
            timeout=25,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="ML service timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ML service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/get-jobs/")
async def get_jobs(
    query: str = Query(..., min_length=2, max_length=100, description="Job search keywords")
):
    """Get jobs from JSearch API"""
    try:
        api_key = os.getenv("JSEARCH_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="JSearch API key not configured")

        response = requests.get(
            "https://jsearch.p.rapidapi.com/search",
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            },
            params={
                "query": query.strip(),
                "num_pages": "1",
                "page": "1"
            },
            timeout=25
        )
        response.raise_for_status()

        jobs = response.json().get("data", [])
        return {
            "jobs": [{
                "title": job.get("job_title", "No title"),
                "company": job.get("employer_name", "Unknown company"),
                "location": f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(', '),
                "description": (job.get("job_description") or "")[:300] + ("..." if len(job.get("job_description", "")) > 300 else ""),
                "apply_link": job.get("job_apply_link") or "#",
                "posted_at": job.get("job_posted_at_datetime_utc", "Unknown")
            } for job in jobs]
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="JSearch API timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"JSearch API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1",
        "services": {
            "ml_service": bool(os.getenv("RENDER_ML_URL")),
            "jsearch_api": bool(os.getenv("JSEARCH_API_KEY"))
        },
        "environment_variables": {
            "RENDER_ML_URL": "configured" if os.getenv("RENDER_ML_URL") else "missing",
            "JSEARCH_API_KEY": "configured" if os.getenv("JSEARCH_API_KEY") else "missing"
        }
    }

# Vercel handler
handler = Mangum(app)