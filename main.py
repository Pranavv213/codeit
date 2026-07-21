import os
import re
import json
import asyncio
import subprocess
from typing import TypedDict, Dict
from pydantic import BaseModel
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

JOBS_FILE = "jobs.json"

# Helper functions for persistent job storage
def load_jobs() -> Dict[str, Dict[str, str]]:
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_job(app_id: str, status: str, url: str = ""):
    jobs = load_jobs()
    jobs[app_id] = {"status": status, "url": url}
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)

class AgentState(TypedDict):
    app_id: str
    prompt: str
    code: str
    deployment_url: str

client = genai.Client()

# ---------------------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------------------

def code_generation_node(state: AgentState) -> AgentState:
    app_id = state["app_id"]
    print(f"\n--- 🤖 STEP 1: Generating code for App ID: {app_id} ---")
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

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(code)

    return {"code": code}


def deployment_node(state: AgentState) -> AgentState:
    app_id = state["app_id"]
    print(f"\n--- 🚀 STEP 2: Deploying App ID: {app_id} to Vercel ---")
    
    deployment_url = ""
    try:
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

        print(f"🎉 App ID [{app_id}] Live URL: {deployment_url}")

        if deployment_url:
            save_job(app_id, "completed", deployment_url)
        else:
            save_job(app_id, "failed", "")

    except Exception as e:
        print(f"⚠️ Deployment Exception: {e}")
        save_job(app_id, "failed", "")

    return {"deployment_url": deployment_url}

# ---------------------------------------------------------------------------
# LangGraph Setup
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("generator", code_generation_node)
workflow.add_node("deployer", deployment_node)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "deployer")
workflow.add_edge("deployer", END)

langgraph_agent = workflow.compile()

def run_pipeline_task(app_id: str, prompt: str):
    initial_state = {"app_id": app_id, "prompt": prompt, "code": "", "deployment_url": ""}
    langgraph_agent.invoke(initial_state)

# ---------------------------------------------------------------------------
# FastAPI Web App
# ---------------------------------------------------------------------------

app = FastAPI(title="Persistent App ID Deployment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DeployRequest(BaseModel):
    app_id: str
    prompt: str

@app.post("/api/deploy")
async def start_deploy(req: DeployRequest, background_tasks: BackgroundTasks):
    save_job(req.app_id, "processing", "")
    background_tasks.add_task(run_pipeline_task, req.app_id, req.prompt)
    return {"message": "Job started", "app_id": req.app_id, "status": "processing"}

@app.get("/api/get-url/{app_id}")
async def get_url(app_id: str):
    jobs = load_jobs()
    if app_id not in jobs:
        raise HTTPException(status_code=404, detail="App ID not found")
    return jobs[app_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)