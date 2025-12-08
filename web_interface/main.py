from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import tempfile
import shutil
from detect import check_image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    results = []
    total_warnings = []

    for file in files:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        try:
            warnings = check_image(temp_path)
            if warnings:
                total_warnings.extend(warnings)
                result = f"{file.filename}: {len(warnings)} violation(s) detected - {'; '.join(warnings)}"
            else:
                result = f"{file.filename}: No violations detected"
            results.append(result)
        finally:
            os.unlink(temp_path)

    return templates.TemplateResponse("results.html", {"request": request, "results": results, "total_warnings": total_warnings})
