from typing import Literal
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role:Literal["system","user","assisant"]
    content:str

