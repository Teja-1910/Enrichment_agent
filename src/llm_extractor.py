import json
import os

from dotenv import load_dotenv
from groq import Groq

from src.models import CompanyIntelligence

load_dotenv()

GROQ_MODEL = "qwen/qwen3.8-27b"

MAX_CONTENT_CHARS = 120_000
MAX_RETRIES = 1


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

Use ONLY information explicitly supported by the supplied
website content.

Do NOT use outside knowledge.
Do NOT guess.
Do NOT fabricate missing information.

============================================================
OUTPUT REQUIREMENTS
============================================================

Return EXACTLY ONE valid JSON object.

Do NOT return:
- Markdown
- Code fences
- Explanations
- Reasoning
- Comments
- Text before the JSON
- Text after the JSON

The JSON object MUST contain exactly these fields:

{{
  "company_overview": "string",
  "target_audience": "string",
  "contact_points": [],
  "leadership_team": [],
  "confidence_score": 0.0
}}

Each leadership_team item MUST contain:

{{
  "name": "string",
  "role": "string",
  "linkedin_url": null
}}

If a LinkedIn URL is not explicitly present in the supplied
website content, use null.

If no public business contact emails are found, use:

"contact_points": []

If no clearly supported leadership information is found, use:

"leadership_team": []

============================================================
1. COMPANY OVERVIEW
============================================================

Provide approximately two concise sentences.

Describe:
- what the company does
- its main product, platform, or service
- the main value it provides

Use only evidence from the supplied content.

============================================================
2. TARGET AUDIENCE / ICP
============================================================

Identify the company's target audience or ideal customer profile.

Look for explicit evidence involving:

- developers
- engineering teams
- startups
- enterprises
- product teams
- agencies
- industries
- business functions
- customer use cases

Keep the result concise.

============================================================
3. CONTACT POINTS
============================================================

Extract only generic/public business email addresses that
actually appear in the supplied website content.

Examples:

contact@company.com
sales@company.com
support@company.com
hello@company.com
info@company.com
press@company.com
careers@company.com

Rules:

- Never invent an email address.
- Never construct an email address.
- Remove duplicates.
- Only include emails supported by the supplied content.

============================================================
4. LEADERSHIP / TEAM
============================================================

Search all supplied pages for clearly supported leadership
or team members.

Look for:

CEO
CTO
CFO
COO
CPO
Chief ...
Founder
Co-founder
President
Vice President
VP
General Counsel
Executive
Leadership
Our Team
Team
Management

For every person included:

- Their name must be explicitly present.
- Their role must be explicitly supported.
- Do not infer their role.
- Do not use outside knowledge.
- Do not duplicate people.

============================================================
5. LINKEDIN URL
============================================================

Only include a LinkedIn URL if an actual LinkedIn URL appears
in the supplied website content and is associated with that
person.

Do NOT:

- construct LinkedIn URLs
- guess usernames
- search external sources

Otherwise:

"linkedin_url": null

============================================================
6. CONFIDENCE SCORE
============================================================

Return a number between 0.0 and 1.0.

The score should reflect how strongly the extracted information
is supported by the supplied website evidence.

Consider:

- company overview evidence
- target audience evidence
- contact information
- leadership information
- number of relevant pages
- consistency of evidence

Do not give a high score merely because there is a lot of text.

============================================================
7. SOURCE HANDLING
============================================================

Ignore:

- navigation menus
- repeated headers
- repeated footers
- cookie notices
- buttons
- boilerplate
- duplicate content

Prioritize meaningful page content.

============================================================
FINAL RULES
============================================================

Before returning the JSON:

1. Use only supplied website evidence.
2. Remove duplicate contacts.
3. Remove duplicate people.
4. Ensure every leadership member has a name and role.
5. Use null when a LinkedIn URL is unavailable.
6. Keep all text concise.
7. Return exactly one valid JSON object.
8. Do not return anything outside the JSON object.

============================================================
CRAWLED WEBSITE CONTENT
============================================================

{website_content}
"""


def build_retry_prompt(
    company_domain: str,
    website_content: str,
) -> str:
    """
    Build a shorter recovery prompt when the first response
    cannot be parsed or validated.
    """

    return f"""
Return ONLY valid JSON.

Analyze the supplied website content for {company_domain}.

Use only information explicitly present in the content.
Do not use outside knowledge.
Do not guess.

Return exactly this structure:

{{
  "company_overview": "string",
  "target_audience": "string",
  "contact_points": [],
  "leadership_team": [],
  "confidence_score": 0.0
}}

Each leadership member must have:

{{
  "name": "string",
  "role": "string",
  "linkedin_url": null
}}

Use null for missing LinkedIn URLs.

Use [] when no contacts or leadership members are supported.

Keep all text concise.

Return ONLY the JSON object.

WEBSITE CONTENT:

{website_content}
"""


def _clean_website_content(
    website_content: str,
) -> str:
    """
    Prevent excessively large website content from being sent
    to the LLM.
    """

    cleaned_content = website_content.strip()

    if len(cleaned_content) <= MAX_CONTENT_CHARS:
        return cleaned_content

    return cleaned_content[:MAX_CONTENT_CHARS]


def _call_llm(
    groq_client: Groq,
    prompt: str,
) -> str:
    """
    Call the Groq model and return the raw response content.
    """

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise company intelligence "
                    "extraction system. "
                    "Use only evidence from the supplied "
                    "website content. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=4096,
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

    return response_content.strip()


def _parse_and_validate(
    response_content: str,
) -> CompanyIntelligence:
    """
    Parse the LLM JSON response and validate it using Pydantic.
    """

    cleaned_response = response_content.strip()

    # Remove accidental Markdown fences if the model adds them.
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]

    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]

    cleaned_response = cleaned_response.removesuffix("```")

    cleaned_response = cleaned_response.strip()

    try:
        parsed_response = json.loads(
            cleaned_response
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


def extract_company_intelligence(
    company_domain: str,
    website_content: str,
) -> CompanyIntelligence:
    """
    Extract structured company intelligence from crawled
    website content using Qwen 3.8 27B through Groq.

    Pydantic is used as the final schema validation layer.
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

    cleaned_content = _clean_website_content(
        website_content
    )

    extraction_prompt = build_company_extraction_prompt(
        company_domain=company_domain,
        website_content=cleaned_content,
    )

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:

            if attempt == 0:
                prompt = extraction_prompt
            else:
                prompt = build_retry_prompt(
                    company_domain=company_domain,
                    website_content=cleaned_content,
                )

            response_content = _call_llm(
                groq_client=groq_client,
                prompt=prompt,
            )

            return _parse_and_validate(
                response_content
            )

        except ValueError as error:

            last_error = error

            if attempt >= MAX_RETRIES:
                break

    raise ValueError(
        f"LLM extraction failed after retry: {last_error}"
    ) from last_error