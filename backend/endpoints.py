# to run: uvicorn backend.endpoints:app --reload --host

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import uuid
import httpx
from typing import Optional
from backend.extreme import evaluate
import traceback

app = FastAPI()

# Directory to temporarily store uploaded videos
UPLOAD_DIR = "./backend/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "ping"}


@app.post("/evaluate/")
async def evaluate_video(
    file: Optional[UploadFile] = File(None),
    fileURL: Optional[str] = Form(None)
):
    try:
        file_path = None
        
        # Check if file upload was provided
        if file and file.filename:
            # Generate a random UUID for the filename
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            file_content = await file.read()
            with open(file_path, "wb") as f:
                f.write(file_content)
        
        # Check if URL was provided
        elif fileURL:
            # Download the file from the URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(fileURL)
                response.raise_for_status()
                file_content = response.content
            
            # Extract file extension from URL or use default
            url_path = fileURL.split('?')[0]  # Remove query parameters
            file_extension = os.path.splitext(url_path)[1] or '.mp4'
            
            # Generate a random UUID for the filename
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save the downloaded file
            with open(file_path, "wb") as f:
                f.write(file_content)
        
        else:
            raise HTTPException(status_code=400, detail="Either file or fileURL must be provided")
        
        # Evaluate the file
        result = evaluate(file_path)
        
        # Create summary string from statistics
        stats = result.get("statistics", {})
        toxic_count = stats.get('toxic_segments', 0)
        total_count = stats.get('total_segments', 1)  # Avoid division by zero
        toxic_percentage = (toxic_count / total_count * 100) if total_count > 0 else 0
        
        # Check if extremist classification is available
        is_extremist_content = stats.get('is_extremist_content')
        extremist_count = stats.get('extremist_segments', 0)
        extremist_ratio = stats.get('extremist_ratio', 0)
        
        # Check if heuristic was used
        segments = result.get("segments", [])

        if is_extremist_content is not None:
            # Extremist classifier is available (or heuristic was used)
            if is_extremist_content:
                summary = f"EXTREMIST CONTENT DETECTED: {extremist_count}/{total_count} segments ({extremist_ratio*100:.1f}%). Avg probability: {stats.get('avg_extremist_probability', 0)*100:.1f}%"
            else:
                summary = f"Non-extremist content. {extremist_count}/{total_count} extremist segments detected ({extremist_ratio*100:.1f}%)."
        else:
            # Fallback to toxicity-based summary
            if toxic_count == 0:
                summary = "No toxic content detected."
            elif toxic_count == 1:
                summary = f"1 toxic segment detected ({toxic_percentage:.1f}% of content). Max toxicity: {stats.get('max_toxicity', 0)*100:.1f}%"
            else:
                summary = f"{toxic_count} toxic segments detected ({toxic_percentage:.1f}% of content). Average toxicity: {stats.get('avg_toxicity', 0)*100:.1f}%, Max: {stats.get('max_toxicity', 0)*100:.1f}%"

        return JSONResponse(content={
            "success": True,
            "result": summary,
            "isExtremist": is_extremist_content,
            "heuristicUsed": heuristic_used,
            "segments": result["segments"],
            "full_text_classification": result.get("full_text_classification"),
            "statistics": result.get("statistics"),
        })
    
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {str(e)}")
    except Exception as e:
        print("Error: " + str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
