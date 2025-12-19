from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import shutil
import os
from project.plate_detect import LicensePlateDetector

app = FastAPI(title='Licences Plate Detect')

# Serve static files
app.mount("/static", StaticFiles(directory="project/static"), name="static")
templates = Jinja2Templates(directory="project/templates")

UPLOAD_DIR = "project/static/uploads"
RESULT_DIR = "project/static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf_threshold: float = Form(0.5), save_interval: int = Form(5)):
    # Save uploaded video
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Output path
    output_path = os.path.join(RESULT_DIR, f"detected_{file.filename}")

    # Run LicensePlateDetector
    detector = LicensePlateDetector(
        video_path=upload_path,
        model_path='project/weights/best_licence_patedt15.pt',
        conf_threshold=conf_threshold,
        save_interval=save_interval
    )
    detector.run(output_path=output_path)  # modify your detector to accept output path

    return FileResponse(output_path, media_type='video/mp4', filename=f"detected_{file.filename}")
