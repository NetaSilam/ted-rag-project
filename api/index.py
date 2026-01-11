from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
import sys

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    try:
        index_path = os.path.join(os.path.dirname(__file__), '..', 'index.html')
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"error": "index.html not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/style.css")
async def get_css():
    """Serve the CSS file"""
    try:
        css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
        if os.path.exists(css_path):
            return FileResponse(css_path, media_type="text/css")
        return JSONResponse({"error": "style.css not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/script.js")
async def get_js():
    """Serve the JavaScript file"""
    try:
        js_path = os.path.join(os.path.dirname(__file__), '..', 'script.js')
        if os.path.exists(js_path):
            return FileResponse(js_path, media_type="application/javascript")
        return JSONResponse({"error": "script.js not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stats")
def stats_endpoint():
    return JSONResponse({
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 15
    })


@app.post("/api/prompt")
async def prompt_endpoint(data: Question):
    try:
        # Import here to catch import errors
        from app.rag import answer_question

        response, context_list, system_prompt, user_prompt = answer_question(data.question)
        return JSONResponse({
            "response": response,
            "context": context_list,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt
            }
        })
    except ImportError as e:
        return JSONResponse({
            "error": f"Import error: {str(e)}",
            "tip": "Check that app/rag.py and dependencies are properly deployed",
            "traceback": str(e.__traceback__)
        }, status_code=200)  # Return 200 so we can see the error in UI
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": f"Error processing question: {str(e)}",
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }, status_code=200)  # Return 200 so we can see the error in UI


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify deployment"""
    import_status = {}
    try:
        from app.rag import answer_question
        import_status["app.rag"] = "OK"
    except Exception as e:
        import_status["app.rag"] = f"FAILED: {str(e)}"

    try:
        from app.utils import load_ted_data
        import_status["app.utils"] = "OK"
    except Exception as e:
        import_status["app.utils"] = f"FAILED: {str(e)}"

    return JSONResponse({
        "status": "running",
        "python_path": sys.path,
        "working_dir": os.getcwd(),
        "imports": import_status
    })