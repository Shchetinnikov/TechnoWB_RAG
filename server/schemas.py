from enum import Enum

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Content(Enum):
    message = Message


class Response(BaseModel):
    status: int
    data: Content
