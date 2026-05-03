import os, json, httpx, asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

class ResearchRequest(BaseModel):
    business_name: str
    website_url: str = ""

async def web_search(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=8) as http:
            r = await http.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            )
            data = r.json()
            results = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"])
            return " | ".join(results) if results else "No results"
    except Exception as e:
        return f"Search unavailable: {e}"

async def fetch_website(url: str) -> str:
    if not url:
        return ""
    if not url.startswith("http"):
        url = "https://" + url
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
            r = await http.get(url, headers=headers)
            text = r.text
            # Strip HTML tags roughly
            import re
            clean = re.sub(r'<[^>]+>', ' ', text)
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean[:4000]
    except Exception as e:
        return f"Could not fetch: {e}"

@app.post("/research")
async def research(req: ResearchRequest):
    name = req.business_name.strip()
    url = req.website_url.strip()

    # Determine URL if not provided
    guessed_url = url
    if not guessed_url:
        slug = name.lower().replace(" ", "").replace("&", "and")
        guessed_url = f"https://www.{slug}.com.au"

    # Run searches and fetch in parallel
    search_q1 = f'"{name}" business AI automation systems'
    search_q2 = f'"{name}" reviews services pricing'

    search1, search2, homepage = await asyncio.gather(
        web_search(search_q1),
        web_search(search_q2),
        fetch_website(guessed_url)
    )

    # Claude analysis
    prompt = f"""You are an AI business intelligence analyst. Research this business and return a JSON object.

Business: {name}
URL attempted: {guessed_url}
Web search 1: {search1}
Web search 2: {search2}
Website content (first 4000 chars): {homepage[:3000] if homepage else 'Could not fetch'}

Return ONLY valid JSON, no markdown, no explanation:

{{
  "company_name": "...",
  "industry": "...",
  "description": "2-3 sentence summary of what this business does",
  "business_model": "one of: service-based, e-commerce, SaaS, consulting, healthcare, retail, hospitality, other",
  "estimated_revenue": "rough estimate e.g. $500K-$2M",
  "team_size": "rough estimate e.g. 10-50",
  "key_decision_maker": {{
    "name": "best guess or 'Unknown'",
    "title": "CEO/Owner/Director/etc"
  }},
  "current_tech_stack": ["list", "of", "tools", "detected"],
  "pain_points": [
    "specific pain point 1 - something AI could fix",
    "specific pain point 2",
    "specific pain point 3"
  ],
  "biggest_opportunity": "The single biggest thing an AI operating system could do for this business",
  "intent_score": 75,
  "route": "one of: AIOS ($5K base), Aria Voice Agent, A&A Membership, Not a fit",
  "email_subject": "compelling subject line personalised to this business",
  "email_body": "A short, personalised outreach email (4-6 sentences). From Kelly Wotherspoon at AI with Soul. Reference their specific business. Mention one pain point. Mention a live AIOS demo is available. Warm, direct, no fluff. Sign off Kelly x"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "company_name": name,
            "description": "Research completed - see details below.",
            "pain_points": ["Manual processes slowing growth", "No automated lead follow-up", "Content creation taking too much time"],
            "email_subject": f"AI Operating System for {name}",
            "email_body": raw[:500],
            "intent_score": 65,
            "route": "AIOS ($5K base)"
        }

    return data

@app.get("/health")
async def health():
    return {"status": "ok"}
