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

# glm-4.6v on the Z.AI coding endpoint: a vision VLM that reads screenshots AND does
# function-calling browser-use can parse. Verified 2026-07-08 the coding endpoint
# (api.z.ai/api/coding/paas/v4) accepts base64 screenshots and works with the coding-plan
# key (the standard paas/v4 endpoint needs a separate balance). Provider swappable via env.
LLM = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "glm-4.6v"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
    api_key=(os.getenv("LLM_API_KEY") or os.getenv("ZAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")),
    temperature=0.2,
)

FIELDS = [
    "business_name", "owner_name", "owner_title",
    "residential_address", "commercial_address", "source", "confidence",
]

TASK = """Find the human owner/manager of the US business "{business}" located in {city}, {state} from the OFFICIAL {state} Secretary of State business registry. Work FAST: return as soon as you have the name + the best address from the registry. Do NOT go to Google or the business website to "verify" — a downstream skip-trace step confirms the home + phone, so extra web research just wastes time.

HARD STOP RULE (most important): the MOMENT you have an owner/agent name and the address that is shown on the entity's detail page, call done() with the JSON. Do NOT open annual reports, statements of information, tax/franchise filings, images of documents, DBA registries, license registries, or ANY additional page to find or reconfirm a residential address. The registry address as-shown is enough — the skip-trace step finds the real home. Opening extra pages is the #1 thing that wastes minutes; never do it once the entity is found.

Steps:
1. You have ALREADY been taken to the official {state} Secretary of State business-entity search page — it is open in front of you right now. Do NOT navigate to any other URL, do NOT type a web address into the address bar, do NOT use a search engine, and do NOT rely on a URL you remember (remembered SoS URLs are often wrong/dead and will fail with a DNS error). Just use the search box ON THE CURRENT PAGE. IMPORTANT search strategy: search the CORE legal name only — strip any location/branch suffix after a dash (e.g. for "SEAPORT MEDSPA - South Boston" search "SEAPORT MEDSPA"). Use the DEFAULT entity-name search mode ("begins with" / entity name), NOT full-text search. If no results, try close variants (with/without LLC, INC).
2. If the search returns MULTIPLE entities, pick the one whose listed address is in or nearest to {city}, {state} (or whose name matches the branch suffix). Open the entity's detail page — ONE page only. Read the managers/members/officers/organizer names, the registered/resident agent, and the addresses shown ON THAT DETAIL PAGE. Do NOT open annual reports, statements of information, filing history, document images, or any linked sub-page — whatever the detail page shows is all you need.
3. Pick the owner: the managing member / organizer / president / manager. If the registered/resident agent is an individual PERSON (not a service company such as "CT Corporation", "Registered Agents Inc", "Northwest", "LegalZoom", "Incfile"), that person is usually a primary owner of a small business — use them.
4. ADDRESSES — use only what is already visible on the detail page; do NOT go looking for a residential address anywhere else. If an individual PERSON's address on the page looks residential (a house, apartment, condo, or a street address with a unit/apt number, e.g. "133 Seaport Blvd Unit 812", "11542 Clearwater St"), put it in residential_address. Put clearly-commercial addresses (registered-agent service company, storefront/plaza/office tower, PO box) in commercial_address. If the detail page shows no residential address, leave residential_address EMPTY and move on — the skip-trace step will find the home. Never invent an address, and never open another page to hunt for one.
5. Return IMMEDIATELY (call done()) once you have the owner name and the address from the detail page. Do not keep browsing, do not re-open the entity, do not double-check on any other page.
5b. NEVER create, write, or read files (no todo.md, no results.md — file actions waste 20-40 seconds each). Keep findings in memory and put the final JSON directly in your done() answer. Where the page allows it, combine multiple actions in one step (e.g. type into the search box AND click Search together).
6. If a page shows a captcha, WAIT a few seconds (an extension auto-solves it); if a Cloudflare Turnstile is stuck call solve_cloudflare once; if the {state} site hard-blocks you (403) even after that, give up on this business and return empty — do NOT go to any other site.
7. NO-RESULTS FALLBACK — applies ONLY when the {state} SoS entity search returned NO matching entity at all after trying name variants. If you DID find the entity, ignore this step entirely and call done(). ({state} official registries ONLY — never Google, Facebook, Yelp, business websites, OpenCorporates, Bizapedia, or ANY other state's website.) When there is truly no SoS match, the business is likely a sole proprietorship / DBA not in the corporate registry. Check these OTHER {state} STATE registries in order, stopping as soon as one gives an owner name: (a) the {state} assumed-name / DBA (fictitious business name) registry, where sole proprietors file their owner name; (b) for a LICENSED TRADE (locksmith, electrician, plumber, HVAC, contractor, cosmetology, etc.) the {state} occupational LICENSE registry — these list the license holder's real name (e.g. in Texas the DPS TOPS / Private Security registry at tops.portal.texas.gov lists locksmith owners). Use ONLY official {state} registry sites — stay within {state}. If none return a name, leave owner_name empty, set confidence low, and STOP — do not search any further sites.

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


# Direct SoS business-search URLs: lets initial_actions skip the first LLM step
# entirely. All 50 states + DC. A few are JS single-page apps (AZ, WA, CA, DC, UT,
# MI, PA, ...) — the headed browser renders them fine even though a raw GET would
# see only the app shell. Five flagged as of Jul 2026 (may fail → business skips):
#   HI (portal mid-migration to dcca.hawaii.gov), LA (coraweb legacy lookup),
#   DC (SPA), GA (WAF), OR (bot-block). Confirm in a browser if leads cluster there.
STATE_SOS_URLS = {
    "AL": "https://arc-sos.state.al.us/CGI/CORPNAME.MBR/INPUT",
    "AK": "https://www.commerce.alaska.gov/cbp/main/search/entities",
    "AZ": "https://ecorp.azcc.gov/EntitySearch/Index",
    "AR": "https://www.ark.org/corp-search/",
    "CA": "https://bizfileonline.sos.ca.gov/search/business",
    "CO": "https://www.sos.state.co.us/biz/BusinessEntityCriteriaExt.do",
    "CT": "https://service.ct.gov/business/s/onlinebusinesssearch",
    "DE": "https://icis.corp.delaware.gov/ecorp/entitysearch/namesearch.aspx",
    "DC": "https://corponline.dlcp.dc.gov/",
    "FL": "https://search.sunbiz.org/Inquiry/CorporationSearch/ByName",
    "GA": "https://ecorp.sos.ga.gov/businesssearch",
    "HI": "https://hbe.dcca.hawaii.gov/",
    "ID": "https://sosbiz.idaho.gov/search/business",
    "IL": "https://apps.ilsos.gov/businessentitysearch/",
    "IN": "https://bsd.sos.in.gov/publicbusinesssearch",
    "IA": "https://sos.iowa.gov/search/business/search.aspx",
    "KS": "https://www.sos.ks.gov/eforms/BusinessEntity/Search.aspx",
    "KY": "https://sosbes.sos.ky.gov/BusSearchNProfile/search.aspx",
    "LA": "https://coraweb.sos.la.gov/commercialsearch/commercialsearch.aspx",
    "ME": "https://apps3.web.maine.gov/nei-sos-icrs/ICRS?MainPage=x",
    "MD": "https://egov.maryland.gov/businessexpress/entitysearch",
    "MA": "https://corp.sec.state.ma.us/corpweb/CorpSearch/CorpSearch.aspx",
    "MI": "https://mibusinessregistry.lara.state.mi.us/search/business",
    "MN": "https://mblsportal.sos.mn.gov/Business/Search",
    "MS": "https://corp.sos.ms.gov/corp/portal/c/page/corpBusinessIdSearch/portal.aspx",
    "MO": "https://www.sos.mo.gov/BusinessEntity/soskb/csearch.asp",
    "MT": "https://biz.sosmt.gov/search",
    "NE": "https://www.nebraska.gov/sos/corp/corpsearch.cgi?nav=search",
    "NV": "https://esos.nv.gov/EntitySearch/OnlineEntitySearch",
    "NH": "https://quickstart.sos.nh.gov/online/BusinessInquire",
    "NJ": "https://www.njportal.com/DOR/BusinessNameSearch/Search/BusinessName",
    "NM": "https://enterprise.sos.nm.gov/search/business",
    "NY": "https://apps.dos.ny.gov/publicInquiry/",
    "NC": "https://www.sosnc.gov/online_services/search/by_title/_Business_Registration",
    "ND": "https://firststop.sos.nd.gov/search/business",
    "OH": "https://businesssearch.ohiosos.gov/",
    "OK": "https://www.sos.ok.gov/corp/corpInquiryFind.aspx",
    "OR": "https://egov.sos.state.or.us/br/pkg_web_name_srch_inq.login",
    "PA": "https://file.dos.pa.gov/search/business",
    "RI": "https://business.sos.ri.gov/CorpWeb/CorpSearch/CorpSearch.aspx",
    "SC": "https://businessfilings.sc.gov/BusinessFiling/Entity/Search",
    "SD": "https://sosenterprise.sd.gov/BusinessServices/Business/FilingSearch.aspx",
    "TN": "https://tnbear.tn.gov/Ecommerce/FilingSearch.aspx",
    "TX": "https://comptroller.texas.gov/taxes/franchise/account-status/search",
    "UT": "https://businessregistration.utah.gov/EntitySearch/OnlineEntitySearch",
    "VT": "https://bizfilings.vermont.gov/online/BusinessInquire/BusinessSearch",
    "VA": "https://cis.scc.virginia.gov/EntitySearch/Index",
    "WA": "https://ccfs.sos.wa.gov/#/AdvancedSearch",
    "WV": "https://apps.wv.gov/sos/businessentitysearch/",
    "WI": "https://apps.dfi.wi.gov/apps/CorpSearch/Search.aspx",
    "WY": "https://wyobiz.wyo.gov/business/filingsearch.aspx",
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
        use_vision=True,      # glm-4.6v (LLM_MODEL) reads screenshots — needed to navigate JS SPAs like CA bizfileonline
        # Force the real OpenAI tools API. glm-4.6v returns clean structured tool_calls
        # via tools=[...] but leaks its native <tool_call> pseudo-XML into content under
        # browser-use's default json_mode → "Could not parse response. Extra data".
        tool_calling_method="function_calling",
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
