import json
from pathlib import Path
from urllib.parse import urlparse

from content_cleaner import clean_webpage_content
from crawler import (
    crawl_relevant_pages,
    discover_relevant_page_urls,
    fetch_page_content,
)
from llm_extractor import extract_company_intelligence
from models import CompanyLead

# =============================================================
# Configuration
# =============================================================

OUTPUT_DIRECTORY = Path("output")

TARGET_COMPANY_DOMAINS = [
    "postman.com",
    "supabase.com",
    "vapi.ai",
]

MAX_LLM_CONTEXT_CHARACTERS = 18000

HOMEPAGE_CHARACTER_LIMIT = 4500

PAGE_CHARACTER_LIMITS = {
    "about": 5500,
    "team": 5500,
    "company": 5000,
    "contact": 4000,
    "pricing": 3500,
    "press": 2000,
    "careers": 1500,
}

PAGE_PRIORITY_KEYWORDS = [
    "about",
    "team",
    "company",
    "contact",
    "pricing",
    "press",
    "careers",
]


# =============================================================
# Page prioritization
# =============================================================

def identify_page_priority(page_url: str) -> int:
    """
    Assign a priority to a webpage based on its URL.

    Lower numbers represent higher relevance for
    company intelligence extraction.
    """

    normalized_url = page_url.lower()

    for priority, keyword in enumerate(
        PAGE_PRIORITY_KEYWORDS
    ):
        if keyword in normalized_url:
            return priority

    return len(PAGE_PRIORITY_KEYWORDS)


def get_page_character_limit(page_url: str) -> int:
    """
    Return the maximum number of characters that should
    be sent to the LLM from a specific page.
    """

    normalized_url = page_url.lower()

    for keyword, character_limit in PAGE_CHARACTER_LIMITS.items():
        if keyword in normalized_url:
            return character_limit

    return 2500


def is_homepage_url(
    page_url: str,
    company_domain: str,
) -> bool:
    """
    Determine whether a URL represents the company's homepage.
    """

    parsed_url = urlparse(page_url)

    page_domain = (
        parsed_url.netloc
        .lower()
        .removeprefix("www.")
    )

    target_domain = (
        company_domain
        .lower()
        .removeprefix("www.")
    )

    return (
        page_domain == target_domain
        and parsed_url.path.rstrip("/") == ""
    )


# =============================================================
# LLM context preparation
# =============================================================

def build_llm_context(
    crawled_pages: dict[str, str],
    company_domain: str,
) -> tuple[str, list[str]]:
    """
    Clean crawled webpage text and build a token-efficient
    context for the LLM.

    Higher-value pages such as About, Team, Contact, and
    Pricing receive more context than lower-value pages.
    """

    homepage_pages = []
    relevant_pages = []

    for page_url, page_text in crawled_pages.items():

        cleaned_page_text = clean_webpage_content(
            page_text
        )

        if not cleaned_page_text:
            continue

        if is_homepage_url(
            page_url=page_url,
            company_domain=company_domain,
        ):
            homepage_pages.append(
                (
                    page_url,
                    cleaned_page_text,
                )
            )
        else:
            relevant_pages.append(
                (
                    identify_page_priority(page_url),
                    page_url,
                    cleaned_page_text,
                )
            )

    relevant_pages.sort(
        key=lambda page: page[0]
    )

    selected_page_sections = []
    pages_analyzed = []

    total_context_characters = 0

    # ---------------------------------------------------------
    # Add homepage first
    # ---------------------------------------------------------

    for page_url, page_text in homepage_pages:

        remaining_characters = (
            MAX_LLM_CONTEXT_CHARACTERS
            - total_context_characters
        )

        if remaining_characters <= 0:
            break

        selected_character_count = min(
            HOMEPAGE_CHARACTER_LIMIT,
            remaining_characters,
        )

        selected_page_text = page_text[
            :selected_character_count
        ]

        selected_page_sections.append(
            
                f"SOURCE URL:\n"
                f"{page_url}\n\n"
                f"PAGE CONTENT:\n"
                f"{selected_page_text}"
            
        )

        pages_analyzed.append(page_url)

        total_context_characters += len(
            selected_page_text
        )

    # ---------------------------------------------------------
    # Add relevant internal pages by priority
    # ---------------------------------------------------------

    for _, page_url, page_text in relevant_pages:

        remaining_characters = (
            MAX_LLM_CONTEXT_CHARACTERS
            - total_context_characters
        )

        if remaining_characters <= 0:
            break

        page_character_limit = (
            get_page_character_limit(page_url)
        )

        selected_character_count = min(
            page_character_limit,
            remaining_characters,
        )

        selected_page_text = page_text[
            :selected_character_count
        ]

        if not selected_page_text:
            continue

        selected_page_sections.append(
            
                f"SOURCE URL:\n"
                f"{page_url}\n\n"
                f"PAGE CONTENT:\n"
                f"{selected_page_text}"
            
        )

        pages_analyzed.append(page_url)

        total_context_characters += len(
            selected_page_text
        )

    combined_website_content = "\n\n".join(
        selected_page_sections
    )

    return (
        combined_website_content,
        pages_analyzed,
    )


# =============================================================
# Failed company result
# =============================================================

def create_failed_company_lead(
    source_urls: list[str],
    pages_analyzed: list[str],
    extraction_warnings: list[str],
) -> CompanyLead:
    """
    Create a consistent empty CompanyLead when the
    enrichment pipeline cannot complete successfully.
    """

    return CompanyLead(
        company_overview="",
        target_audience="",
        contact_points=[],
        leadership_team=[],
        confidence_score=0.0,
        source_urls=source_urls,
        pages_analyzed=pages_analyzed,
        crawl_status="failed",
        extraction_warnings=extraction_warnings,
    )


# =============================================================
# Company enrichment
# =============================================================

def enrich_company(
    company_domain: str,
) -> CompanyLead:
    """
    Run the complete company lead-enrichment pipeline
    for one company domain.
    """

    print("\n" + "=" * 60)
    print(
        f"Processing: {company_domain}"
    )
    print("=" * 60)

    extraction_warnings = []

    homepage_url = (
        f"https://{company_domain}"
    )

    # =========================================================
    # STEP 1 — Fetch homepage
    # =========================================================

    print(
        "\n[1/5] Fetching homepage..."
    )

    homepage_html = fetch_page_content(
        company_domain=company_domain
    )

    if not homepage_html:

        extraction_warnings.append(
            "Homepage could not be fetched."
        )

        return create_failed_company_lead(
            source_urls=[],
            pages_analyzed=[],
            extraction_warnings=extraction_warnings,
        )

    print(
        f"Homepage HTML: "
        f"{len(homepage_html):,} characters"
    )

    # =========================================================
    # STEP 2 — Discover relevant pages
    # =========================================================

    print(
        "\n[2/5] Discovering relevant pages..."
    )

    discovered_page_urls = (
        discover_relevant_page_urls(
            page_html=homepage_html,
            homepage_url=homepage_url,
        )
    )

    print(
        f"Relevant pages discovered: "
        f"{len(discovered_page_urls)}"
    )

    for page_url in discovered_page_urls:
        print(
            f"  - {page_url}"
        )

    # =========================================================
    # STEP 3 — Crawl pages
    # =========================================================

    print(
        "\n[3/5] Crawling pages..."
    )

    crawled_pages = crawl_relevant_pages(
        company_domain=company_domain,
        discovered_page_urls=discovered_page_urls,
    )

    print(
        f"Successfully crawled: "
        f"{len(crawled_pages)} pages"
    )

    expected_page_count = (
        len(discovered_page_urls) + 1
    )

    if len(crawled_pages) < expected_page_count:

        extraction_warnings.append(
            "One or more discovered pages "
            "could not be crawled."
        )

    # =========================================================
    # STEP 4 — Clean and optimize content
    # =========================================================

    print(
        "\n[4/5] Cleaning and optimizing "
        "webpage content..."
    )

    website_content, pages_analyzed = (
        build_llm_context(
            crawled_pages=crawled_pages,
            company_domain=company_domain,
        )
    )

    print(
        f"LLM context characters: "
        f"{len(website_content):,}"
    )

    print(
        f"Pages selected for LLM analysis: "
        f"{len(pages_analyzed)}"
    )

    for page_url in pages_analyzed:
        print(
            f"  - {page_url}"
        )

    if not website_content:

        extraction_warnings.append(
            "No usable webpage content "
            "was available."
        )

        return create_failed_company_lead(
            source_urls=list(
                crawled_pages.keys()
            ),
            pages_analyzed=[],
            extraction_warnings=extraction_warnings,
        )

    # =========================================================
    # STEP 5 — LLM extraction
    # =========================================================

    print(
        "\n[5/5] Extracting company "
        "intelligence with LLM..."
    )

    try:

        company_intelligence = (
            extract_company_intelligence(
                company_domain=company_domain,
                website_content=website_content,
            )
        )

    except Exception as error:  # noqa: BLE001

        extraction_warnings.append(
            f"LLM extraction failed: {error}"
        )

        return create_failed_company_lead(
            source_urls=list(
                crawled_pages.keys()
            ),
            pages_analyzed=pages_analyzed,
            extraction_warnings=extraction_warnings,
        )

    # =========================================================
    # Build final company result
    # =========================================================

    company_lead = CompanyLead(
        **company_intelligence.model_dump(),

        source_urls=list(
            crawled_pages.keys()
        ),

        pages_analyzed=pages_analyzed,

        crawl_status="success",

        extraction_warnings=extraction_warnings,
    )

    return company_lead


# =============================================================
# Save final results
# =============================================================

def save_all_company_leads(
    company_leads: list[CompanyLead],
) -> Path:
    """
    Save enrichment results for all target companies
    into one output.json file.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIRECTORY
        / "output.json"
    )

    output_data = [
        company_lead.model_dump()
        for company_lead in company_leads
    ]

    output_file.write_text(
        json.dumps(
            output_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_file


# =============================================================
# Main
# =============================================================

def main() -> None:
    """
    Run the lead-enrichment pipeline for all target companies.

    Each company is processed independently so that a failure
    for one domain does not stop the remaining companies.
    """

    company_leads = []

    print(
        "\n"
        + "#" * 60
    )

    print(
        "AI LEAD ENRICHMENT AGENT"
    )

    print(
        "#" * 60
    )

    print(
        f"\nTarget companies: "
        f"{len(TARGET_COMPANY_DOMAINS)}"
    )

    for company_domain in TARGET_COMPANY_DOMAINS:

        try:

            company_lead = enrich_company(
                company_domain=company_domain
            )

        except Exception as error:  # noqa: BLE001

            print(
                f"\nUnexpected error while processing "
                f"{company_domain}: {error}"
            )

            company_lead = (
                create_failed_company_lead(
                    source_urls=[],
                    pages_analyzed=[],
                    extraction_warnings=[
                        f"Unexpected pipeline error: {error}"
                    ],
                )
            )

        company_leads.append(
            company_lead
        )

    # =========================================================
    # Save combined output
    # =========================================================

    output_file = save_all_company_leads(
        company_leads=company_leads
    )

    # =========================================================
    # Final summary
    # =========================================================

    print(
        "\n"
        + "#" * 60
    )

    print(
        "PIPELINE COMPLETE"
    )

    print(
        "#" * 60
    )

    successful_companies = sum(
        1
        for company_lead in company_leads
        if company_lead.crawl_status == "success"
    )

    failed_companies = (
        len(company_leads)
        - successful_companies
    )

    print(
        f"\nSuccessful companies: "
        f"{successful_companies}"
    )

    print(
        f"Failed companies: "
        f"{failed_companies}"
    )

    print(
        f"Output saved to: "
        f"{output_file}"
    )

    print(
        "\nFinal structured output:"
    )

    print(
        json.dumps(
            [
                company_lead.model_dump()
                for company_lead in company_leads
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()