from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import answer_question

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
    return {"message": "TED RAG API is running", "status": "healthy"}


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


handler = app
