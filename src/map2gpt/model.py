
from enum import Enum
from pydantic import BaseModel

from typing import List 

from dataclasses import dataclass

class Role(str, Enum):
    USER:str='user'
    SYSTEM:str='system'
    ASSITANT:str='assistant'

class Message(BaseModel):
    role:Role
    content:str

@dataclass
class ExtractedFeatures:
    document_id:str 
    chunks:List[str]
    embeddings:List[List[float]]
    name:str

class YouTubeTranscriotionLanguage(str, Enum):
    EN:str='English'
    FR:str='French'
