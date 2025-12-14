from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import os

print("="*50)
print("Starting TED RAG API...")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current dir: {os.listdir('.')}")
print(f"LLMOD_API_KEY exists: {bool(os.getenv('LLMOD_API_KEY'))}")
print(f"PINECONE_API_KEY exists: {bool(os.getenv('PINECONE_API_KEY'))}")
print("="*50)

try:
    from rag import answer_question
    print("✅ Successfully imported answer_question")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    print(traceback.format_exc())
    raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    try:
        response, context_list, system_prompt, user_prompt = answer_question(q.question)
        return {
            "response": response,
            "context": context_list,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt
            }
        }
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
def get_stats():
    return {
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 15
    }

print("✅ App initialized successfully")
