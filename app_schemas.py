# app_schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel

class A2APart(BaseModel):
    text: str

class A2AMessage(BaseModel):
    role: Literal["user", "agent"]
    parts: List[A2APart]

class A2AArtifact(BaseModel):
    artifact_id: str
    name: str
    parts: List[A2APart]

class A2ATaskRequest(BaseModel):
    task_id: str
    context_id: str
    message: A2AMessage
    artifacts: Optional[List[A2AArtifact]] = None

class A2ATaskResponse(BaseModel):
    task_id: str
    context_id: str
    status: Literal["working", "completed", "failed"]
    output_message: Optional[A2AMessage] = None
    artifacts: Optional[List[A2AArtifact]] = None
    error_message: Optional[str] = None

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str

class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    url: str
    skills: List[AgentSkill]