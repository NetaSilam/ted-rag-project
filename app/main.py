import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

print("=" * 50, flush=True)
print("🚀 STARTING TED RAG API", flush=True)
print("=" * 50, flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Current directory: {os.getcwd()}", flush=True)
print(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}", flush=True)
print(f"Files in current dir: {os.listdir('.')}", flush=True)
print("=" * 50, flush=True)

# Try to import rag module with error handling
try:
    print("📦 Attempting to import answer_question from rag...", flush=True)
    from rag import answer_question
    print("✅ Successfully imported answer_question", flush=True)
except Exception as e:
    print(f"❌ ERROR importing answer_question: {e}", flush=True)
    print(f"Exception type: {type(e)}", flush=True)
    import traceback
    traceback.print_exc()
    # Create a dummy function so the app can still start
    def answer_question(q):
        return "Import failed", [], "", ""

print("🔧 Initializing FastAPI app...", flush=True)
app = FastAPI()

# Add CORS for Vercel
print("🌐 Adding CORS middleware...", flush=True)
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
    print("📍 Root endpoint called", flush=True)
    return {"message": "TED RAG API is running", "status": "healthy"}


@app.post("/api/prompt")
def ask_question(q: Question):
    print(f"📝 Question received: {q.question}", flush=True)
    try:
        response, context_list, system_prompt, user_prompt = answer_question(q.question)
        print("✅ Question answered successfully", flush=True)
        return {
            "response": response,
            "context": context_list,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt
            }
        }
    except Exception as e:
        print(f"❌ ERROR in ask_question: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/stats")
def get_stats():
    print("📊 Stats endpoint called", flush=True)
    return {
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 15
    }


# Startup event to verify app is ready
@app.on_event("startup")
async def startup_event():
    print("=" * 50, flush=True)
    print("🎉 FastAPI app startup event triggered", flush=True)
    print("=" * 50, flush=True)


handler = app

print("✅ App initialization complete", flush=True)
print("=" * 50, flush=True)

# Add this for Railway deployment
if __name__ == "__main__":
    print("🏃 Running uvicorn directly (if __name__ == '__main__')", flush=True)
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"🔌 Starting server on 0.0.0.0:{port}", flush=True)
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="debug")
