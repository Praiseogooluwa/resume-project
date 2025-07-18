from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import requests
import os
import io
import fitz  # PyMuPDF
from typing import Dict, List

# Initialize FastAPI with explicit docs settings
app = FastAPI(
    title="Resume Matcher API",
    description="AI-powered job matching system",
    version="2.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    expose_headers=["X-OpenAPI-Schema"],
    max_age=600
)

# Custom OpenAPI schema generation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=[{"url": "https://your-app-name.vercel.app", "description": "Production"}]
    )
    
    # Add error responses to schema
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["responses"].update({
                "400": {"description": "Validation Error"},
                "500": {"description": "Internal Server Error"},
                "502": {"description": "Service Unavailable"}
            })
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    """Secure PDF text extraction with validation"""
    try:
        # Read first 5MB to check size
        pdf_data = uploaded_file.file.read(5_000_000)
        remaining_bytes = uploaded_file.file.read(1)
        if remaining_bytes:
            return "Error: PDF exceeds 5MB limit (max 5MB allowed)"
            
        pdf_stream = io.BytesIO(pdf_data)
        try:
            with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)
        except fitz.FileDataError:
            return "Error: Invalid or corrupted PDF file"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@app.post(
    "/match-jobs/",
    response_model=Dict[str, List[Dict]],
    summary="Match resume to jobs",
    responses={
        200: {"description": "Successful matching"},
        400: {"description": "Invalid input"},
        502: {"description": "ML service unavailable"}
    }
)
async def match_jobs(
    file: UploadFile = File(..., description="PDF resume file (max 5MB)"),
    query: str = Form(..., min_length=2, description="Job search query")
) -> Dict:
    """
    Processes uploaded resume PDF and returns top job matches using AI.
    - Supports only PDF files under 5MB
    - Requires at least 2-character search query
    """
    # File validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are accepted")

    resume_text = extract_text_from_pdf(file)
    if not resume_text or "Error" in resume_text:
        raise HTTPException(400, detail=resume_text.replace("Error: ", ""))

    # ML service integration
    try:
        ml_service_url = os.getenv("https://resume-ml-api.onrender.com")
        if not ml_service_url:
            raise RuntimeError("ML service URL not configured")

        response = requests.post(
            f"{ml_service_url.strip('/')}/predict",
            json={"resume_text": resume_text, "query": query.strip()},
            timeout=12,  # Slightly above Vercel's 10s limit
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        raise HTTPException(504, detail="ML service timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, detail=f"ML service error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")

@app.get(
    "/get-jobs/",
    response_model=Dict[str, List[Dict]],
    summary="Fetch live job listings",
    responses={
        200: {"description": "Successful job fetch"},
        400: {"description": "Invalid query"},
        502: {"description": "JSearch API unavailable"}
    }
)
async def get_jobs(
    query: str = Query(..., min_length=2, max_length=100, description="Job search keywords")
) -> Dict:
    """
    Retrieves live job listings from JSearch API.
    - Requires 2-100 character search query
    - Returns first page of results
    """
    try:
        api_key = os.getenv("JSEARCH_API_KEY")
        if not api_key:
            raise RuntimeError("API key not configured")

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
            timeout=10
        )
        response.raise_for_status()
        
        jobs = response.json().get("data", [])
        return {
            "jobs": [{
                "title": job.get("job_title", "No title"),
                "company": job.get("employer_name", "Unknown company"),
                "location": ", ".join(filter(None, [
                    job.get("job_city"),
                    job.get("job_country")
                ])),
                "description": (job.get("job_description") or "")[:300] + ("..." if len(job.get("job_description", "")) > 300 else ""),
                "apply_link": job.get("job_apply_link") or "#"
            } for job in jobs]
        }
        
    except requests.exceptions.Timeout:
        raise HTTPException(504, detail="JSearch API timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, detail=f"JSearch API error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")

# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "healthy", "version": app.version}

# Ensure OpenAPI schema is generated at startup
@app.on_event("startup")
async def startup_event():
    custom_openapi()
