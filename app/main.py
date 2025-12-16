import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import answer_question

app = FastAPI()

# Add CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "THIS IS A NEW DEPLOY", "status": "ok"}


@app.post("/api/prompt")
def ask_question(q: Question):
    response, context_list, system_prompt, user_prompt = answer_question(q.question)
    return {
        "response": response,
        "context": context_list,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }


@app.get("/api/stats")
def get_stats():
    return {
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 15
    }


@app.get("/__version")
def version():
    return {
        "version": "2025-01-12-category-debug-v3"
    }


handler = app

# Add this for Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
