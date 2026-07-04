"""Owner Finder service — a browser-use agent that finds a US business owner's
name + residential address by browsing the state Secretary of State registry
and the open web. Called by the n8n "Owner Cell Pipeline 2" workflow.

Captcha handling: the CapSolver browser extension is loaded into a HEADED
Chromium (run under xvfb) and auto-solves reCAPTCHA / Turnstile / hCaptcha
transparently. A `solve_cloudflare` Controller action is registered as an
explicit fallback the agent can call for a Turnstile widget.

POST /find-owner  {"business": "...", "state": "MA", "token": "..."}
Returns the JSON shape the n8n Owner Schema parser expects.
"""
import asyncio
import inspect
import json
import os
import re
import shutil
import tempfile
import time

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from browser_use import Agent, Controller, ActionResult, BrowserSession, BrowserProfile
try:
    from browser_use.browser.types import Page
except Exception:  # fall back to the raw playwright type
    from playwright.async_api import Page
try:
    from browser_use.llm import ChatOpenAI
except ImportError:  # browser-use 0.3.x uses LangChain chat models
    from langchain_openai import ChatOpenAI

EXT_DIR = os.getenv("EXT_DIR", "/opt/capext")
CAP_KEY = os.getenv("CAPSOLVER_KEY", "")
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
RUN_TIMEOUT = int(os.getenv("RUN_TIMEOUT", "540"))

LLM = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "glm-5.2"),
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
5. If a page shows a captcha (reCAPTCHA / Cloudflare Turnstile / hCaptcha), just WAIT a few seconds — an extension auto-solves it. If a Cloudflare Turnstile widget is stuck, call the solve_cloudflare action once. If a site hard-blocks you ("you have been blocked" / 403), move on to another source (OpenCorporates.com, Bizapedia, the business's own website).

When done, your final answer must be ONLY this JSON (empty strings for anything not found):
{{"business_name": "{business}", "owner_name": "", "owner_title": "", "residential_address": "", "commercial_address": "", "source": "", "confidence": "low|medium|high"}}"""

controller = Controller()


@controller.action("Solve a Cloudflare Turnstile widget on the current page and inject the token")
async def solve_cloudflare(page: Page) -> ActionResult:
    if not CAP_KEY:
        return ActionResult(extracted_content="capsolver key not set", include_in_memory=True)
    try:
        sitekey = await page.get_attribute(".cf-turnstile", "data-sitekey")
    except Exception:
        sitekey = None
    if not sitekey:
        return ActionResult(extracted_content="no turnstile widget on page", include_in_memory=True)

    task = {"type": "AntiTurnstileTaskProxyLess", "websiteURL": page.url, "websiteKey": sitekey}
    meta = {}
    for attr, key in (("data-action", "action"), ("data-cdata", "cdata")):
        try:
            v = await page.get_attribute(".cf-turnstile", attr)
        except Exception:
            v = None
        if v:
            meta[key] = v
    if meta:
        task["metadata"] = meta

    try:
        r = requests.post("https://api.capsolver.com/createTask",
                          json={"clientKey": CAP_KEY, "task": task}, timeout=30).json()
        if r.get("errorId"):
            return ActionResult(extracted_content=f"capsolver createTask: {r.get('errorDescription')}")
        tid, token = r["taskId"], None
        for _ in range(40):
            time.sleep(3)
            res = requests.post("https://api.capsolver.com/getTaskResult",
                                json={"clientKey": CAP_KEY, "taskId": tid}, timeout=30).json()
            if res.get("errorId"):
                return ActionResult(extracted_content=f"capsolver result: {res.get('errorDescription')}")
            if res.get("status") == "ready":
                token = res["solution"]["token"]
                break
        if not token:
            return ActionResult(extracted_content="turnstile solve timed out")
        await page.evaluate(
            """(t) => {
                let el = document.querySelector('input[name="cf-turnstile-response"]');
                if (!el) { el = document.createElement('input'); el.type='hidden';
                           el.name='cf-turnstile-response'; document.forms[0]?.appendChild(el); }
                el.value = t;
            }""",
            token,
        )
        return ActionResult(extracted_content="injected turnstile token", include_in_memory=True)
    except Exception as e:
        return ActionResult(extracted_content=f"solve_cloudflare error: {e}")


def _build_profile(user_data_dir: str) -> BrowserProfile:
    return BrowserProfile(
        headless=False,  # headed → the CapSolver extension works; xvfb provides the display
        user_data_dir=user_data_dir,  # non-None → launch_persistent_context → extensions load
        window_size={"width": 1920, "height": 1080},
        args=[
            f"--disable-extensions-except={EXT_DIR}",
            f"--load-extension={EXT_DIR}",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
        ],
    )


async def _shutdown(session) -> None:
    for name in ("stop", "close", "kill"):
        fn = getattr(session, name, None)
        if fn:
            try:
                res = fn()
                if inspect.isawaitable(res):
                    await res
                return
            except Exception:
                continue


app = FastAPI()


class Req(BaseModel):
    business: str
    state: str
    token: str = ""


@app.get("/health")
async def health():
    return {"ok": True, "ext": os.path.isfile(os.path.join(EXT_DIR, "manifest.json"))}


@app.post("/find-owner")
async def find_owner(r: Req):
    svc = os.getenv("SVC_TOKEN", "")
    if svc and r.token != svc:
        raise HTTPException(status_code=401, detail="bad token")

    tmp = tempfile.mkdtemp(prefix="bu-")
    session = BrowserSession(browser_profile=_build_profile(tmp))
    agent = Agent(
        task=TASK.format(business=r.business, state=r.state),
        llm=LLM,
        controller=controller,
        browser_session=session,
        enable_memory=False,  # mem0 needs an OpenAI embeddings key we don't use
        use_vision=False,     # Z.AI coding endpoint is text-only, rejects screenshots
    )
    final = ""
    try:
        history = await asyncio.wait_for(agent.run(max_steps=MAX_STEPS), timeout=RUN_TIMEOUT)
        final = history.final_result() or ""
    except asyncio.TimeoutError:
        print(f"agent timeout for {r.business}", flush=True)
    except Exception as e:
        print(f"agent error for {r.business}: {e}", flush=True)
    finally:
        await _shutdown(session)
        shutil.rmtree(tmp, ignore_errors=True)

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
