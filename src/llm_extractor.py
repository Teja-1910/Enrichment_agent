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

    The prompt instructs the model to use only evidence from the
    crawled website pages and to avoid hallucinating missing data.
    """

    return f"""
You are a careful AI company intelligence extraction agent.

Your task is to analyze the supplied public website content for:

COMPANY DOMAIN:
{company_domain}

The website content contains multiple crawled pages. Each page is
preceded by its SOURCE URL.

Your job is to extract reliable company intelligence from ALL of
the supplied pages.

============================================================
CORE RULE
============================================================

Use ONLY information explicitly supported by the supplied website
content.

Do NOT use outside knowledge.

Do NOT guess.

Do NOT infer a person's identity, job title, email address, or
LinkedIn URL when the evidence is not present.

If information cannot be reliably supported by the supplied content,
return the appropriate empty value.

============================================================
1. COMPANY OVERVIEW
============================================================

Create a concise overview of approximately TWO sentences.

Describe:
- what the company does
- its main product/platform/service
- the main value it provides

Do not copy large sections of the website.

Do not include unsupported claims.

============================================================
2. TARGET AUDIENCE / ICP
============================================================

Identify the company's target audience or ideal customer profile.

Look across ALL supplied pages for evidence such as:
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

============================================================
3. CONTACT POINTS
============================================================

Extract GENERIC/PUBLIC BUSINESS EMAIL ADDRESSES found in the
supplied website content.

Examples include:

contact@company.com
sales@company.com
support@company.com
hello@company.com
info@company.com
press@company.com
careers@company.com

Do NOT invent email addresses.

Do NOT construct an email address from the company domain.

Do NOT include personal/private email addresses unless the website
clearly presents the address as a generic public business contact.

Remove duplicate email addresses.

============================================================
4. KEY LEADERSHIP / TEAM
============================================================

Search ALL supplied pages carefully for leadership and team
information.

Look especially for sections or text containing:

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

Careers, company, about, press, and other pages may contain
leadership information, so do not inspect only one page.

For every person included, the supplied content must explicitly
support BOTH:

1. Their name
2. Their role/title

Do NOT invent team members.

Do NOT infer leadership from outside knowledge.

Do NOT include a person's name without a supported role.

============================================================
5. LINKEDIN URL
============================================================

Include a LinkedIn URL ONLY when an actual LinkedIn URL is present
in the supplied website content and is associated with that person.

Valid examples:

https://www.linkedin.com/in/example

Do NOT construct a LinkedIn URL from a person's name.

Do NOT guess a LinkedIn username.

Do NOT search external sources.

If an actual LinkedIn URL is not present in the supplied content,
return:

null

This is important: a missing LinkedIn URL does NOT mean the person
does not have LinkedIn. It only means that the URL was not
discoverable from the supplied website content.

============================================================
6. CONFIDENCE SCORE
============================================================

Return a confidence score between 0.0 and 1.0.

The score should represent how strongly the extracted company
intelligence is supported by the supplied website evidence.

Consider:

- quality of company overview evidence
- quality of target-audience evidence
- availability of public contact information
- availability of explicit leadership/team information
- number and relevance of supplied pages
- consistency of information across pages

Use a lower score when important information is missing or weakly
supported.

Do not automatically assign a high score simply because the website
contains a lot of text.

Do not automatically assign a low score merely because a LinkedIn
URL is missing.

============================================================
7. SOURCE HANDLING
============================================================

The supplied content may contain navigation menus, repeated
headers, footers, cookie notices, buttons, and other boilerplate.

Ignore irrelevant navigation and repeated boilerplate.

Give greater attention to meaningful page content.

When the same information appears multiple times, treat it as one
piece of evidence rather than duplicating it.

============================================================
FINAL RULES
============================================================

Before producing the result:

1. Inspect ALL supplied pages.
2. Extract only supported information.
3. Remove duplicate contacts.
4. Remove duplicate people.
5. Ensure every leadership member has a supported name and role.
6. Only include LinkedIn URLs that actually appear in the supplied
   content.
7. Never fabricate missing information.
8. Return ONLY the structured JSON object required by the schema.

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
    Send cleaned website content to GPT-OSS 120B and return
    validated structured company intelligence.
    """

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Check the .env file."
        )

    if not website_content.strip():
        raise ValueError(
            "No website content was supplied "
            "to the LLM."
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
                    "Extract only evidence-supported information "
                    "from the supplied website content. "
                    "Follow the JSON schema exactly."
                ),
            },
            {
                "role": "user",
                "content": extraction_prompt,
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "company_intelligence",
                "strict": True,
                "schema": CompanyIntelligence.model_json_schema(),
            },
        },
    )

    response_content = (
        response.choices[0].message.content
    )

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

    return CompanyIntelligence.model_validate(
        parsed_response
    )