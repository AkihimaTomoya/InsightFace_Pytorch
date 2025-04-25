from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
import cv2
import numpy as np
import time
from typing import List
from face_verify import faceRec
import uvicorn

# Create a fresh FastAPI instance
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Configuration
config = {
    "show_bbox": True,
    "show_label": True
}

# Initialize face recognition
camera = faceRec()
last_recognition_time = 0
recognition_interval = 2


# Serve the HTML page
@app.get("/")
async def get_index():
    try:
        with open("templates/index.html", "r") as f:
            content = f.read()
            return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<html><body><h1>Error: Template file not found</h1></body></html>")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    # Send configuration
    await websocket.send_json({"type": "config", "data": config})

    try:
        while True:
            data = await websocket.receive_text()
            await process_frame(websocket, data)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# Process video frames
async def process_frame(websocket: WebSocket, data: str):
    global last_recognition_time
    current_time = time.time()

    try:
        # Decode the image
        if ',' in data:
            _, encoded = data.split(',', 1)
        else:
            encoded = data
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Face locations update - always
        face_locations = camera.get_face_locations(frame)
        await websocket.send_json({"type": "face_locations", "data": face_locations})

        # Recognition update - every 2 seconds
        if current_time - last_recognition_time >= recognition_interval:
            recognition_data = camera.recognize_faces(frame)
            await websocket.send_json({"type": "recognition_data", "data": recognition_data})
            last_recognition_time = current_time

    except Exception as e:
        print(f"Frame processing error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
