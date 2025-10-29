# chat_service.py
import os, uvicorn
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import UploadFile, File
import cv2
import numpy as np
import mediapipe as mp

load_dotenv()
app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.5)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")
genai.configure(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.post("/analyze/snapshot")
async def analyze_snapshot(file: UploadFile = File(...)):
    """
    Analyzes a single snapshot using the global MediaPipe pose instance.
    """
    try:
        # 1. Read and decode the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # 2. Process the image using the global pose object
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            raise HTTPException(status_code=400, detail="Could not detect a person in the image.")

        # 3. Format landmarks for the frontend
        landmarks_data = []
        all_landmarks = results.pose_landmarks.landmark
        
        for id_value, lm in enumerate(all_landmarks):
            landmarks_data.append({
                "id": id_value,
                "name": mp_pose.PoseLandmark(id_value).name,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        # 4. Return the data
        return {"landmarks": landmarks_data}

    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI errors
    except Exception as e:
        print(f"Snapshot Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.post("/chatbot")
async def chat_with_bot(request: ChatRequest):
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
        response = model.generate_content(request.message)
        if response.text:
            return {"reply": response.text}
        else:
            raise HTTPException(status_code=500, detail="Failed to get a valid response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) # Note the different port