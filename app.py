"""Owner Finder service — browser-use agent that finds a business owner's
name + residential address via the state Secretary of State registry and
open web. Called by the n8n "Owner Cell Pipeline 2" workflow.

POST /find-owner  {"business": "...", "state": "MA", "token": "..."}
Returns the JSON shape the n8n Owner Schema parser expects.
"""
import json
import os
import re

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from browser_use import Agent
try:
    from browser_use.llm import ChatOpenAI
except ImportError:  # browser-use 0.3.x uses LangChain chat models
    from langchain_openai import ChatOpenAI

LLM = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "glm-4.6"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
    api_key=os.environ["ZAI_API_KEY"],
    temperature=0.2,
)

FIELDS = [
    "business_name", "owner_name", "owner_title",
    "residential_address", "commercial_address", "source", "confidence",
]

TASK = """Find the real human owner of the US business "{business}" in state {state}.

Steps:
1. Go to the {state} Secretary of State business-entity search (search the web for it if needed) and look up the business. Try close variants of the name (with/without LLC, INC).
2. From the filing, capture: officers/members/managers/organizer names, the registered agent, the registered/principal office address.
3. The owner is usually the managing member / organizer / president. If the registered agent is a person (not a service company like "Registered Agents Inc"), they are often the owner of small businesses.
4. A residential address is often the organizer's or registered agent's address on small LLC filings, or search the owner's name + city on the open web for their home address. Only report a residential address you actually found — NEVER guess.
5. If the SoS site blocks you or has a captcha you cannot pass, try OpenCorporates.com or Bizapedia for the same entity.

When done, your final answer must be ONLY this JSON (empty strings for anything not found):
{{"business_name": "{business}", "owner_name": "", "owner_title": "", "residential_address": "", "commercial_address": "", "source": "", "confidence": "low|medium|high"}}"""

app = FastAPI()


class Req(BaseModel):
    business: str
    state: str
    token: str = ""


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/find-owner")
async def find_owner(r: Req):
    svc = os.getenv("SVC_TOKEN", "")
    if svc and r.token != svc:
        raise HTTPException(status_code=401, detail="bad token")

    agent = Agent(task=TASK.format(business=r.business, state=r.state), llm=LLM)
    try:
        history = await agent.run(max_steps=int(os.getenv("MAX_STEPS", "30")))
        final = history.final_result() or ""
    except Exception as e:  # surface agent failures as empty result, not 500
        final = ""
        print(f"agent error for {r.business}: {e}", flush=True)

    data = {}
    m = re.search(r"\{.*\}", final, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    out = {k: str(data.get(k, "") or "") for k in FIELDS}
    out["business_name"] = out["business_name"] or r.business
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
