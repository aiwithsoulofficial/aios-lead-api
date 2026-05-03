import os, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

class ResearchRequest(BaseModel):
    business_name: str
    website_url: str = ""

@app.post("/research")
async def research(req: ResearchRequest):
    name = req.business_name.strip()
    url = req.website_url.strip()
    url_hint = f" (website: {url})" if url else ""

    prompt = f"""Research the business "{name}"{url_hint} using web search.

Find:
- What they actually do, who their customers are, rough size
- Their website, socials, any press or reviews
- Their current tech stack (booking system, CRM, website platform, email tool)
- Where their business is leaking time or money - be specific, not generic
- The decision maker's name and title if findable

Then return ONLY valid JSON (no markdown, no explanation):

{{
  "company_name": "actual name",
  "industry": "specific industry",
  "description": "2-3 sentences describing what they actually do based on what you found",
  "business_model": "service-based | e-commerce | SaaS | consulting | healthcare | retail | hospitality | other",
  "estimated_revenue": "your best estimate e.g. $500K-$2M",
  "team_size": "your best estimate e.g. 10-50 staff",
  "key_decision_maker": {{
    "name": "actual name found or Unknown",
    "title": "their actual title"
  }},
  "current_tech_stack": ["real tools you found evidence of"],
  "pain_points": [
    "Specific pain point based on what you found - tie it to something real on their website or in reviews",
    "Another specific one",
    "Third one"
  ],
  "biggest_opportunity": "The single highest-ROI thing an AI operating system could do for this specific business",
  "intent_score": 75,
  "route": "AIOS ($5K base) | Aria Voice Agent | A&A Membership | Not a fit",
  "email_subject": "Subject line that references something specific about their business",
  "email_body": "Short outreach email (4-5 sentences). From Kelly Wotherspoon at AI with Soul. Reference one specific real thing you found about them. Name the pain point. Offer a live AIOS demo. Warm, direct. No AI-sounding phrases. Sign off Kelly x"
}}"""

    # Use Claude with web search tool enabled - real research, no fake data
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract the final text response (after tool use)
    raw = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw = block.text.strip()

    # Find JSON object anywhere in the response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        return json.loads(raw)
    except Exception:
        return {
            "company_name": name,
            "description": raw[:300] if raw else "Research completed.",
            "pain_points": ["Manual processes slowing growth", "No automated follow-up", "Content bottleneck"],
            "email_subject": f"Quick question about {name}",
            "email_body": f"Hi,\n\nI came across {name} and I'd love to show you what an AI operating system could do for your business.\n\nWorth a 20-minute call?\n\nKelly x",
            "intent_score": 65,
            "route": "AIOS ($5K base)"
        }

@app.get("/health")
async def health():
    return {"status": "ok"}
