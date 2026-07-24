import os
import json
import uuid
import httpx
import uvicorn
from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse  # <--- Updated import
from fastapi.middleware.cors import CORSMiddleware

from app_schemas import A2ATaskRequest, A2AMessage, A2APart, A2ATaskResponse

JOBS_FILE = "jobs.json"
CODE_AGENT_URL = os.getenv("CODE_AGENT_URL", "http://localhost:8001/a2a/tasks")
DEPLOY_AGENT_URL = os.getenv("DEPLOY_AGENT_URL", "http://localhost:8002/a2a/tasks")

def load_jobs() -> Dict[str, Dict[str, str]]:
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading {JOBS_FILE}: {e}")
            return {}
    return {}

def save_job(app_id: str, status: str, url: str = ""):
    jobs = load_jobs()
    jobs[app_id] = {"status": status, "url": url}
    
    # Write atomically via temp file to avoid partial write corruptions
    temp_file = f"{JOBS_FILE}.tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2)
        os.replace(temp_file, JOBS_FILE)
    except Exception as e:
        print(f"⚠️ Error saving job {app_id}: {e}")

def run_a2a_orchestration_pipeline(app_id: str, prompt: str):
    context_id = f"ctx-{app_id}"
    print(f"\n[Orchestrator] Starting A2A pipeline for Context ID: {context_id}")

    try:
        with httpx.Client(timeout=180.0) as client:
            # Step 1: Dispatch Code Generation
            task_1_req = A2ATaskRequest(
                task_id=f"task-gen-{uuid.uuid4().hex[:6]}",
                context_id=context_id,
                message=A2AMessage(role="user", parts=[A2APart(text=prompt)])
            )

            print(f"[Orchestrator] Dispatching code generation task to Agent 1...")
            resp_1 = client.post(CODE_AGENT_URL, json=task_1_req.model_dump())
            agent_1_res = A2ATaskResponse(**resp_1.json())

            if agent_1_res.status == "failed":
                print(f"⚠️ Agent 1 Failure: {agent_1_res.error_message}")
                save_job(app_id, "failed", "")
                return

            code_artifact = agent_1_res.artifacts[0]

            # Step 2: Dispatch Deployment
            task_2_req = A2ATaskRequest(
                task_id=f"task-dep-{uuid.uuid4().hex[:6]}",
                context_id=context_id,
                message=A2AMessage(role="user", parts=[A2APart(text="Deploy generated artifact")]),
                artifacts=[code_artifact]
            )

            print(f"[Orchestrator] Dispatching deployment task to Agent 2...")
            resp_2 = client.post(DEPLOY_AGENT_URL, json=task_2_req.model_dump())
            agent_2_res = A2ATaskResponse(**resp_2.json())

            if agent_2_res.status == "failed":
                print(f"⚠️ Agent 2 Failure: {agent_2_res.error_message}")
                save_job(app_id, "failed", "")
                return

            live_url = agent_2_res.output_message.parts[0].text
            print(f"🎉 Pipeline Succeeded! Live URL: {live_url}")
            save_job(app_id, "completed", live_url)

    except Exception as e:
        print(f"⚠️ Orchestrator Pipeline Exception: {e}")
        save_job(app_id, "failed", "")

app = FastAPI(title="Multi-Agent A2A Deployment Orchestrator")

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

# --- SERVE FRONTEND AT ROOT PATH ---
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    if not os.path.exists("frontend.html"):
        raise HTTPException(status_code=404, detail="frontend.html not found")
    return FileResponse("frontend.html")

@app.post("/api/deploy")
async def start_deploy(req: DeployRequest, background_tasks: BackgroundTasks):
    save_job(req.app_id, "processing", "")
    background_tasks.add_task(run_a2a_orchestration_pipeline, req.app_id, req.prompt)
    return {"message": "Job started", "app_id": req.app_id, "status": "processing"}

@app.get("/api/get-url/{app_id}")
async def get_url(app_id: str):
    jobs = load_jobs()
    if app_id not in jobs:
        raise HTTPException(status_code=404, detail="App ID not found")
    return jobs[app_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)