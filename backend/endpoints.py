# to run: uvicorn backend.endpoints:app --reload --host

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid

app = FastAPI()

# Directory to temporarily store uploaded videos
UPLOAD_DIR = "./backend/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "ping"}

@app.post("/evaluate/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Generate a random UUID for the filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)
        

        # TODO: DO SOME PROCESSING HERE

        # segments = whisper(file_path)
        # for s in segments:
        #    analyze(s)
        # result = ....

        return JSONResponse(content={
            "hate": True,
            "result": {}
        })
    except Exception as e:
        print("Error: " + e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
