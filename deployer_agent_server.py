# deployer_agent_server.py
import re
import uvicorn
import subprocess
from typing import TypedDict
from fastapi import FastAPI
from langgraph.graph import StateGraph, END

from app_schemas import (
    AgentCard, AgentSkill, A2ATaskRequest, 
    A2ATaskResponse, A2AMessage, A2APart
)

app = FastAPI(title="A2A Deployment Agent")

class DeployerState(TypedDict):
    code: str
    deployment_url: str

def deployment_node(state: DeployerState) -> DeployerState:
    code = state["code"]
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(code)

    deployment_url = ""
    cmd = "npx vercel deploy --prod --yes"
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120
    )
    output = result.stdout

    matches = re.findall(r'https://[a-zA-Z0-9-]+\.vercel\.app', output)
    if matches:
        deployment_url = matches[-1]
    else:
        for line in reversed(output.splitlines()):
            line_str = line.strip()
            if line_str.startswith("https://") and "vercel" in line_str:
                deployment_url = line_str.split()[0]
                break

    if not deployment_url:
        raise RuntimeError("Vercel deployment output did not yield a valid URL.")

    return {"deployment_url": deployment_url}

builder = StateGraph(DeployerState)
builder.add_node("deploy", deployment_node)
builder.add_edge("deploy", END)
builder.set_entry_point("deploy")
deployer_graph = builder.compile()

AGENT_CARD = AgentCard(
    name="Deployment Agent",
    description="Deploys HTML code artifacts directly to Vercel.",
    version="1.0.0",
    url="http://localhost:8002/a2a/tasks",
    skills=[AgentSkill(id="vercel-deploy", name="Deployer", description="Deploys code via Vercel CLI")]
)

@app.get("/.well-known/agent-card.json")
def get_card():
    return AGENT_CARD

@app.post("/a2a/tasks", response_model=A2ATaskResponse)
def handle_task(request: A2ATaskRequest):
    print(f"\n[Agent 2: Deployer] Received task ID: {request.task_id}")
    try:
        if not request.artifacts or not request.artifacts[0].parts:
            raise ValueError("No code artifact passed to deployment agent")

        code = request.artifacts[0].parts[0].text
        res = deployer_graph.invoke({"code": code, "deployment_url": ""})

        return A2ATaskResponse(
            task_id=request.task_id,
            context_id=request.context_id,
            status="completed",
            output_message=A2AMessage(
                role="agent", 
                parts=[A2APart(text=res["deployment_url"])]
            )
        )
    except Exception as e:
        return A2ATaskResponse(
            task_id=request.task_id,
            context_id=request.context_id,
            status="failed",
            error_message=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)