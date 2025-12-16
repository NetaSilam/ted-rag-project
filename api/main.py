from fastapi import FastAPI
from api import prompt, stats

app = FastAPI()

# Use include_router instead of mount
app.include_router(prompt.app, prefix="/api/prompt", tags=["prompt"])
app.include_router(stats.app, prefix="/api/stats", tags=["stats"])
