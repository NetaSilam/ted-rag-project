from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.rag import answer_question

app = APIRouter()


class Question(BaseModel):
    question: str


@app.post("/")
async def prompt(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse({"error": 'Missing "question" field'}, status_code=400)

    response, context_list, system_prompt, user_prompt = answer_question(question)
    return JSONResponse({
        "response": response,
        "context": context_list,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    })
