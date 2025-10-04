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
        res_tuple = evaluate(file_path)
        res = res_tuple[0]  # whisper result (dict with 'segments')
        audio = res_tuple[1]  # path to normalized audio

        # Load audio as mono 16kHz
        import librosa
        y_all, sr = librosa.load(audio, sr=16000, mono=True)

        # Get segments from whisper result
        segments = res["segments"]

        # Import the intonation module
        from backend.testing.transcript_proccessing.intonation_wav2vec2_module import process_segments

        # Run intonation/emotion extraction
        results = process_segments(y_all, sr, segments)

        print(results)

        return JSONResponse(content={
            "success": True,
            "result": "success",
            "segments": segments,
            "intonation_results": results,
        })
    
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {str(e)}")
    except Exception as e:
        print("Error: " + str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
