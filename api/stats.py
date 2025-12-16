from fastapi import APIRouter
from fastapi.responses import JSONResponse

app = APIRouter()


@app.get("/")
def stats():
    return JSONResponse({
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 5
    })
