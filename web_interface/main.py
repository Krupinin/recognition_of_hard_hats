from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import tempfile
import shutil
from detect import check_image
import gettext

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_translations(lang_code):
    """Get translations for the given language"""
    try:
        locale_path = os.path.join('locale', lang_code, 'LC_MESSAGES', 'messages.mo')
        if os.path.exists(locale_path):
            return gettext.translation('messages', localedir='locale', languages=[lang_code])
    except Exception:
        pass
    return None

def get_text_function(lang_code):
    """Get translation function for templates"""
    trans = get_translations(lang_code)
    if trans:
        return trans.gettext
    else:
        return str  # fallback to original string

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    lang = request.cookies.get('language', 'en')
    trans_func = get_text_function(lang)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "_": trans_func
    })

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

    lang = request.cookies.get('language', 'en')
    trans_func = get_text_function(lang)
    return templates.TemplateResponse("results.html", {
        "request": request,
        "results": results,
        "total_warnings": total_warnings,
        "_": trans_func
    })

@app.post("/set_language")
async def set_language(request: Request, language: str = Form(...)):
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="language", value=language, httponly=True)
    return response
