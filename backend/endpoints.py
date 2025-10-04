# to run: uvicorn backend.endpoints:app --reload --host

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os

app = FastAPI()

# Directory to temporarily store uploaded videos
UPLOAD_DIR = "./backend/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "ping"}

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        file_content = await file.read()  # Read the file content
        with open(file_path, "wb") as f:
            print(file_content)
            f.write(file_content)  # Write the content to the file
        return JSONResponse(content={"file_path": file_path})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# TODO: Replace with actual ML model processing

@app.post("/process/")
async def process_video(file_path: str):
    # PIPELINE
    # transcript            : Map<Timestamp, String> (per sentence, process foreach from this point)
    # hate speech detection : Map<Timestamp, Float> (per sentence, per model)
    # OUTPUT
    # return : Map<Timestamp, 2Tuple<String, Map<String, Float>>>
    #          (map sentence -> (transcript, map model -> sentiment score))

    return {"message": "Processing not yet implemented."}