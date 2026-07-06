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

TASK = """Find the human owner/manager of the US business "{business}" located in {city}, {state} from the OFFICIAL {state} Secretary of State business registry. Work FAST: return as soon as you have the name + the best address from the registry. Do NOT go to Google or the business website to "verify" — a downstream skip-trace step confirms the home + phone, so extra web research just wastes time.

Steps:
1. Go to the {state} Secretary of State business-entity search and look up the business. IMPORTANT search strategy: search the CORE legal name only — strip any location/branch suffix after a dash (e.g. for "SEAPORT MEDSPA - South Boston" search "SEAPORT MEDSPA"). Use the DEFAULT entity-name search mode ("begins with" / entity name), NOT full-text search. If no results, try close variants (with/without LLC, INC).
2. If the search returns MULTIPLE entities, pick the one whose listed address is in or nearest to {city}, {state} (or whose name matches the branch suffix). Open the entity's detail page (and its latest ANNUAL REPORT if it's one click away — annual reports often list officer/director RESIDENCE addresses). Capture the managers/members/officers/organizer names, the registered/resident agent, and every address shown.
3. Pick the owner: the managing member / organizer / president / manager. If the registered/resident agent is an individual PERSON (not a service company such as "CT Corporation", "Registered Agents Inc", "Northwest", "LegalZoom", "Incfile"), that person is usually a primary owner of a small business — use them.
4. RESIDENTIAL ADDRESS — report an address in residential_address when the agent/officer is an individual PERSON AND the address looks residential: a house, apartment, condo, or a street address with a unit/apt/suite-in-a-residential-building number (e.g. "133 Seaport Blvd Unit 812", "11542 Clearwater St"). On a small owner-operated business the resident agent's / officer's address IS usually their home — so DO report it, you do NOT need to confirm it on the web. Only leave residential_address empty if the ONLY addresses are a commercial registered-agent service company, an obvious business storefront/plaza/office tower, or a PO box. Put clearly-commercial addresses (principal office, storefront) in commercial_address. Never invent an address.
5. Return IMMEDIATELY once you have the owner name and the registry address. Do not keep browsing to double-check.
5b. NEVER create, write, or read files (no todo.md, no results.md — file actions waste 20-40 seconds each). Keep findings in memory and put the final JSON directly in your done() answer. Where the page allows it, combine multiple actions in one step (e.g. type into the search box AND click Search together).
6. If a page shows a captcha, WAIT a few seconds (an extension auto-solves it); if a Cloudflare Turnstile is stuck call solve_cloudflare once; if a site hard-blocks you (403), try OpenCorporates.com or Bizapedia for the same entity.

When done, your final answer must be ONLY this JSON (empty strings for anything not found):
{{"business_name": "{business}", "owner_name": "", "owner_title": "", "residential_address": "", "commercial_address": "", "source": "", "confidence": "low|medium|high"}}"""

class OwnerOut(BaseModel):
    business_name: str = ""
    owner_name: str = ""
    owner_title: str = ""
    residential_address: str = ""
    commercial_address: str = ""
    source: str = ""
    confidence: str = ""


# output_model forces the done action to take these fields as typed parameters,
# so the LLM can't return an empty text answer (glm-5-turbo did exactly that)
controller = Controller(output_model=OwnerOut)


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


# Direct SoS business-search URLs: lets initial_actions skip the first LLM step entirely
STATE_SOS_URLS = {
    "MA": "https://corp.sec.state.ma.us/corpweb/CorpSearch/CorpSearch.aspx",
    "NH": "https://quickstart.sos.nh.gov/online/BusinessInquire",
    "CT": "https://service.ct.gov/business/s/onlinebusinesssearch",
    "RI": "https://business.sos.ri.gov/CorpWeb/CorpSearch/CorpSearch.aspx",
    "NY": "https://apps.dos.ny.gov/publicInquiry/",
    "FL": "https://search.sunbiz.org/Inquiry/CorporationSearch/ByName",
    "TX": "https://comptroller.texas.gov/taxes/franchise/account-status/search",
    "CA": "https://bizfileonline.sos.ca.gov/search/business",
}


def _build_profile(user_data_dir: str) -> BrowserProfile:
    return BrowserProfile(
        headless=False,  # headed → the CapSolver extension works; xvfb provides the display
        user_data_dir=user_data_dir,  # non-None → launch_persistent_context → extensions load
        window_size={"width": 1920, "height": 1080},
        minimum_wait_page_load_time=0.25,
        wait_between_actions=0.2,
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
    city: str = ""
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
    kw = dict(
        task=TASK.format(business=r.business, state=r.state, city=r.city or "an unknown city"),
        llm=LLM,
        controller=controller,
        browser_session=session,
        enable_memory=False,  # mem0 needs an OpenAI embeddings key we don't use
        use_vision=False,     # Z.AI coding endpoint is text-only, rejects screenshots
    )
    sos_url = STATE_SOS_URLS.get(r.state.upper().strip())
    if sos_url:
        # open the registry before the agent's first LLM call — saves a whole step
        kw["initial_actions"] = [{"go_to_url": {"url": sos_url}}]
    try:
        agent = Agent(**kw, flash_mode=True)  # skip eval/thinking output for speed
    except Exception:
        agent = Agent(**kw)  # older browser-use without flash_mode
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
