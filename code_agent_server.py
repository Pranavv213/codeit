# code_agent_server.py
import uuid
import uvicorn
from typing import TypedDict
from fastapi import FastAPI
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END

from app_schemas import (
    AgentCard, AgentSkill, A2ATaskRequest, 
    A2ATaskResponse, A2AMessage, A2APart, A2AArtifact
)

app = FastAPI(title="A2A Code Generation Agent")
client = genai.Client()

class CodeAgentState(TypedDict):
    prompt: str
    code: str

def code_generation_node(state: CodeAgentState) -> CodeAgentState:
    prompt = state["prompt"]
    system_instruction = (
        "You are an expert frontend developer. Write a complete, standalone, production-ready "
        "single-file index.html application (including all internal CSS and JavaScript) based on the prompt. "
        "Return ONLY valid HTML code block wrapped in ```html ... ```."
    )

    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=f"User Request: {prompt}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    content = response.text

    if "```html" in content:
        code = content.split("```html")[1].split("```")[0].strip()
    elif "```" in content:
        code = content.split("```")[1].split("```")[0].strip()
    else:
        code = content.strip()

    return {"code": code}

builder = StateGraph(CodeAgentState)
builder.add_node("generate_code", code_generation_node)
builder.add_edge(START, "generate_code")
builder.add_edge("generate_code", END)
code_graph = builder.compile()

AGENT_CARD = AgentCard(
    name="Code Generation Agent",
    description="Generates production-ready HTML/JS/CSS code.",
    version="1.0.0",
    url="http://localhost:8001/a2a/tasks",
    skills=[AgentSkill(id="code-gen", name="Code Generator", description="Writes index.html web apps")]
)

@app.get("/.well-known/agent-card.json")
def get_card():
    return AGENT_CARD

@app.post("/a2a/tasks", response_model=A2ATaskResponse)
def handle_task(request: A2ATaskRequest):
    print(f"\n[Agent 1: CodeGen] Received task ID: {request.task_id}")
    try:
        user_prompt = request.message.parts[0].text
        res = code_graph.invoke({"prompt": user_prompt, "code": ""})

        return A2ATaskResponse(
            task_id=request.task_id,
            context_id=request.context_id,
            status="completed",
            output_message=A2AMessage(role="agent", parts=[A2APart(text="Code generated successfully")]),
            artifacts=[
                A2AArtifact(
                    artifact_id=f"art-{uuid.uuid4().hex[:6]}",
                    name="index.html",
                    parts=[A2APart(text=res["code"])]
                )
            ]
        )
    except Exception as e:
        return A2ATaskResponse(
            task_id=request.task_id,
            context_id=request.context_id,
            status="failed",
            error_message=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)