import json
import os

from dotenv import load_dotenv
from groq import Groq

from src.models import CompanyIntelligence

load_dotenv()

GROQ_MODEL = "qwen/qwen3.8-27b"


def build_company_extraction_prompt(
    company_domain: str,
    website_content: str,
) -> str:
    """
    Build a focused extraction prompt for company intelligence.

    The model must use only evidence present in the crawled
    website content and must not invent missing information.
    """

    return f"""
You are a precise AI company intelligence extraction agent.

Analyze the supplied public website content for:

COMPANY DOMAIN:
{company_domain}

The content contains multiple crawled website pages.
Each page may be preceded by its SOURCE URL.

Your task is to extract reliable company intelligence from
ALL supplied pages.

============================================================
CORE RULE
============================================================

Use ONLY information explicitly supported by the supplied
website content.

Do NOT use outside knowledge.

Do NOT guess.

Do NOT infer information that is not supported by the content.

If information is missing or cannot be reliably supported,
return the appropriate empty value defined by the schema.

============================================================
1. COMPANY OVERVIEW
============================================================

Create a concise overview.

Describe:
- what the company does
- its main product, platform, or service
- the main value it provides

Keep the overview concise.

Do not copy large sections of the website.

Do not include unsupported claims.

============================================================
2. TARGET AUDIENCE / ICP
============================================================

Identify the company's target audience or ideal customer profile.

Look across ALL supplied pages for explicit evidence such as:

- developers
- engineering teams
- startups
- enterprises
- product teams
- agencies
- specific industries
- business functions
- customer use cases

Prefer explicit descriptions from the website.

Combine evidence from multiple pages when appropriate.

Keep the answer concise.

============================================================
3. CONTACT POINTS
============================================================

Extract GENERIC/PUBLIC BUSINESS EMAIL ADDRESSES found directly
in the supplied website content.

Examples:

contact@company.com
sales@company.com
support@company.com
hello@company.com
info@company.com
press@company.com
careers@company.com

Rules:

- Do NOT invent email addresses.
- Do NOT construct an email address from the company domain.
- Do NOT include unsupported email addresses.
- Do NOT include personal/private email addresses unless the
  website clearly presents them as public business contacts.
- Remove duplicate email addresses.
- Include only relevant public business contacts.

============================================================
4. KEY LEADERSHIP / TEAM
============================================================

Search ALL supplied pages for leadership and team information.

Look especially for:

- CEO
- CTO
- CFO
- COO
- CPO
- Chief ...
- Founder
- Co-founder
- President
- Vice President
- VP
- General Counsel
- Executive
- Leadership
- Our Team
- Team
- Management

For every person included, the supplied content must explicitly
support BOTH:

1. Their name
2. Their role/title

Rules:

- Do NOT invent team members.
- Do NOT use outside knowledge.
- Do NOT infer a person's role.
- Do NOT include a person's name without a supported role.
- Remove duplicate people.
- Include only clearly supported leadership/team members.

============================================================
5. LINKEDIN URL
============================================================

Include a LinkedIn URL ONLY when an actual LinkedIn URL appears
in the supplied website content and is associated with that person.

Valid example:

https://www.linkedin.com/in/example

Rules:

- Do NOT construct a LinkedIn URL from a person's name.
- Do NOT guess a LinkedIn username.
- Do NOT search external sources.
- If an actual LinkedIn URL is not present in the supplied
  content, return null.

A missing LinkedIn URL does NOT mean the person does not have
LinkedIn. It only means the URL was not discoverable from the
supplied website content.

============================================================
6. CONFIDENCE SCORE
============================================================

Return a confidence score between 0.0 and 1.0.

The score should represent how strongly the extracted information
is supported by the supplied website evidence.

Consider:

- quality of company overview evidence
- quality of target-audience evidence
- availability of public contact information
- availability of explicit leadership/team information
- number and relevance of supplied pages
- consistency of information across pages

Use a lower score when important information is missing or weakly
supported.

Do not automatically assign a high score simply because the
website contains a lot of text.

Do not automatically assign a low score merely because a LinkedIn
URL is missing.

============================================================
7. SOURCE HANDLING
============================================================

The supplied content may contain:

- navigation menus
- repeated headers
- repeated footers
- cookie notices
- buttons
- boilerplate
- duplicate content

Ignore irrelevant navigation and repeated boilerplate.

Give greater attention to meaningful page content.

When the same information appears multiple times, treat it as
one piece of evidence rather than duplicating it.

============================================================
8. OUTPUT SIZE RULES
============================================================

Keep the response concise.

- company_overview: approximately 2 concise sentences.
- target_audience: concise description.
- contact_points: only relevant public business emails.
- leadership_team: only clearly supported people.
- Do not repeat information.
- Do not include explanations outside the requested fields.
- Do not include analysis or reasoning.
- Do not include markdown.
- Return ONLY the JSON object required by the schema.

============================================================
FINAL VALIDATION RULES
============================================================

Before producing the result:

1. Inspect ALL supplied pages.
2. Extract only supported information.
3. Remove duplicate contacts.
4. Remove duplicate people.
5. Ensure every leadership member has both a supported name
   and supported role.
6. Only include LinkedIn URLs that actually appear in the
   supplied content.
7. Never fabricate missing information.
8. Keep the response concise.
9. Follow the supplied JSON schema exactly.
10. Return ONLY the structured JSON object.

============================================================
CRAWLED WEBSITE CONTENT
============================================================

{website_content}
"""


def extract_company_intelligence(
    company_domain: str,
    website_content: str,
) -> CompanyIntelligence:
    """
    Send cleaned website content to Qwen 3.8 27B through Groq
    and return validated structured company intelligence.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Check the environment variables or Streamlit Secrets."
        )

    if not website_content.strip():
        raise ValueError(
            "No website content was supplied to the LLM."
        )

    groq_client = Groq(
        api_key=groq_api_key,
    )

    extraction_prompt = build_company_extraction_prompt(
        company_domain=company_domain,
        website_content=website_content,
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise company intelligence "
                    "extraction system. "
                    "Use only evidence from the supplied website "
                    "content. "
                    "Do not use outside knowledge. "
                    "Follow the JSON schema exactly. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": extraction_prompt,
            },
        ],
        temperature=0,
        reasoning_effort="none",
        max_completion_tokens=4096,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "company_intelligence",
                "strict": True,
                "schema": CompanyIntelligence.model_json_schema(),
            },
        },
    )

    if not response.choices:
        raise ValueError(
            "The LLM returned no choices."
        )

    response_content = response.choices[0].message.content

    if not response_content:
        raise ValueError(
            "The LLM returned an empty response."
        )

    try:
        parsed_response = json.loads(
            response_content
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "The LLM returned invalid JSON."
        ) from error

    try:
        return CompanyIntelligence.model_validate(
            parsed_response
        )

    except Exception as error:
        raise ValueError(
            "The LLM response did not match the expected "
            "company intelligence schema."
        ) from error