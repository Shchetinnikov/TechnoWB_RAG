from fastapi import FastAPI

from .schemas import Message
from .ml_service.rag_init import rag

app = FastAPI(
    title="Server"
)

@app.post("/user_message")
def process_user_message(data: Message):
    assistant_message = rag.ask(data.content)
    return {"status": 200,
            "data": {
                "role": "assistant",
                "content": assistant_message
            }}