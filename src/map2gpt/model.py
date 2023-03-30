
from enum import Enum
from pydantic import BaseModel

from typing import List 

from numpy import ndarray
from dataclasses import dataclass

class Role(str, Enum):
    USER:str='user'
    SYSTEM:str='system'
    ASSITANT:str='assistant'

class Message(BaseModel):
    role:Role
    content:str

class IndexResponse(BaseModel):
    answer:str 
    questions:str 
    source_chunks:List[str]

@dataclass
class ExtractedFeatures:
    chunks:List[str]
    embeddings:ndarray 
    name:str
    description:str