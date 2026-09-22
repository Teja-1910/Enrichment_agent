import os
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------
# Browser configuration
# ---------------------------------------------------------

PAGE_TIMEOUT = 30000
POST_LOAD_WAIT = 2000
MENU_WAIT = 700

# We deliberately keep the crawl bounded.
# The goal is company intelligence, not crawling an entire site.
MAX_RELEVANT_PAGES = 8

def launch_browser(playwright):
    """Launch Chromium locally or use system Chromium in deployment."""

    chromium_path = os.getenv("CHROMIUM_PATH")

    if chromium_path:
        return playwright.chromium.launch(
            executable_path=chromium_path,
            headless=True,
        )

    return playwright.chromium.launch(
        headless=True,
    )


# ---------------------------------------------------------
# Page relevance configuration
# ---------------------------------------------------------
#
# These keywords are chosen around the information required
# by the assignment:
#
# - Company Overview
# - Target Audience / ICP
# - Contact Points
# - Leadership / Team
# - LinkedIn URLs if discoverable
#
# Leadership/company/contact pages therefore receive
# stronger priority than generic customer or resource pages.
# ---------------------------------------------------------

PAGE_PRIORITY_KEYWORDS = {
    # Highest priority: directly useful for leadership,
    # company identity, and contact extraction.
    "leadership": 100,
    "team": 100,
    "people": 100,
    "management": 100,
    "executive": 100,
    "founder": 100,
    "founders": 100,

    "about": 95,
    "about-us": 95,
    "company": 95,

    "contact": 90,
    "contact-us": 90,

    # Strong supporting evidence.
    "careers": 80,
    "career": 80,

    # Useful for ICP and company understanding.
    "customers": 60,
    "customer": 60,
    "solutions": 55,
    "platform": 50,
    "products": 50,
    "product": 50,

    # Useful but less important than the above.
    "pricing": 45,
    "press": 35,
    "news": 30,
    "resources": 20,
    "blog": 15,
}


# Navigation controls that may open a hamburger menu.
MENU_KEYWORDS = [
    "menu",
    "navigation",
    "nav",
    "open menu",
    "open navigation",
]


# Sections that commonly contain nested links.
EXPANDABLE_NAVIGATION_KEYWORDS = [
    "solutions",
    "platform",
    "resources",
    "products",
    "product",
    "company",
    "developers",
    "use cases",
]


# ---------------------------------------------------------
# Homepage retrieval
# ---------------------------------------------------------


def fetch_page_content(
    company_domain: str,
) -> str:
    """
    Fetch the company's homepage HTML using Playwright.

    Playwright is used because modern websites may rely on
    JavaScript to render navigation and page content.

    If navigation times out after usable HTML has already
    been received, the available HTML is returned instead
    of failing the complete pipeline.
    """

    homepage_url = f"https://{company_domain}"

    with sync_playwright() as playwright:

        browser = launch_browser(playwright)

        page = browser.new_page()

        try:
            try:
                page.goto(
                    homepage_url,
                    wait_until="domcontentloaded",
                    timeout=PAGE_TIMEOUT,
                )

            except Exception as error:  # noqa: BLE001
                print(
                    f"Navigation warning for "
                    f"{homepage_url}: {error}"
                )

                current_page_html = page.content()

                if len(current_page_html) > 1000:
                    print(
                        "Usable HTML was received despite "
                        "the navigation timeout."
                    )

                    return current_page_html

                print(
                    "No usable HTML was received."
                )

                return ""

            page.wait_for_timeout(
                POST_LOAD_WAIT
            )

            return page.content()

        except Exception as error:  # noqa: BLE001
            print(
                f"Failed to fetch "
                f"{homepage_url}: {error}"
            )

            return ""

        finally:
            browser.close()


# ---------------------------------------------------------
# URL handling
# ---------------------------------------------------------


def get_normalized_domain(
    website_url: str,
) -> str:
    """
    Return a normalized hostname without the www prefix.
    """

    return (
        urlparse(website_url)
        .netloc
        .lower()
        .removeprefix("www.")
    )


def is_internal_url(
    absolute_url: str,
    homepage_url: str,
) -> bool:
    """
    Check whether a URL belongs to the same company website.

    www.example.com and example.com are treated as the same
    domain.
    """

    homepage_domain = get_normalized_domain(
        homepage_url
    )

    page_domain = get_normalized_domain(
        absolute_url
    )

    return page_domain == homepage_domain


def normalize_internal_url(
    page_url: str,
    homepage_url: str,
) -> str | None:
    """
    Convert a discovered href into a clean absolute URL.

    Ignore:
    - mailto:
    - tel:
    - javascript:
    - fragment-only links
    - external domains
    """

    if not page_url:
        return None

    page_url = page_url.strip()

    if page_url.startswith(
        (
            "mailto:",
            "tel:",
            "javascript:",
            "#",
        )
    ):
        return None

    absolute_url = urljoin(
        homepage_url,
        page_url,
    )

    parsed_url = urlparse(
        absolute_url
    )

    if parsed_url.scheme not in {
        "http",
        "https",
    }:
        return None

    if not is_internal_url(
        absolute_url=absolute_url,
        homepage_url=homepage_url,
    ):
        return None

    # Remove fragments because they do not represent
    # separate pages.
    clean_url = absolute_url.split("#")[0]

    # Remove trailing slash for consistent comparison.
    if clean_url != (
        f"{parsed_url.scheme}://"
        f"{parsed_url.netloc}"
    ):
        clean_url = clean_url.rstrip("/")

    return clean_url


# ---------------------------------------------------------
# HTML link extraction
# ---------------------------------------------------------


def collect_page_links(
    page_html: str,
    homepage_url: str,
) -> list[tuple[str, str]]:
    """
    Extract internal links and their visible text.

    Returns:
        [
            (normalized_url, link_text),
            ...
        ]
    """

    soup = BeautifulSoup(
        page_html,
        "html.parser",
    )

    discovered_links = []

    for link in soup.find_all(
        "a",
        href=True,
    ):
        normalized_url = normalize_internal_url(
            page_url=link["href"],
            homepage_url=homepage_url,
        )

        if not normalized_url:
            continue

        link_text = link.get_text(
            " ",
            strip=True,
        ).lower()

        discovered_links.append(
            (
                normalized_url,
                link_text,
            )
        )

    return discovered_links


# ---------------------------------------------------------
# Dynamic navigation
# ---------------------------------------------------------


def find_menu_button(page):
    """
    Find a likely hamburger/navigation button.

    The function first checks accessibility attributes
    such as aria-label/title and then falls back to detecting
    icon-based buttons near the top of the page.
    """

    buttons = page.locator(
        "button, [role='button']"
    )

    button_count = buttons.count()

    for button_index in range(
        min(button_count, 50)
    ):

        try:
            button = buttons.nth(
                button_index
            )

            if not button.is_visible(
                timeout=500
            ):
                continue

            aria_label = (
                button.get_attribute(
                    "aria-label"
                )
                or ""
            ).lower()

            button_text = (
                button.inner_text(
                    timeout=500
                )
                or ""
            ).lower()

            title = (
                button.get_attribute(
                    "title"
                )
                or ""
            ).lower()

            combined_text = (
                f"{aria_label} "
                f"{button_text} "
                f"{title}"
            )

            # Best case: accessibility metadata tells us
            # that this is the menu button.
            if any(
                keyword in combined_text
                for keyword in MENU_KEYWORDS
            ):
                return button

            # Hamburger buttons frequently contain only
            # an SVG/icon and no text.
            try:
                svg_count = button.locator(
                    "svg"
                ).count()

                if svg_count == 0:
                    continue

                bounding_box = button.bounding_box()

                if not bounding_box:
                    continue

                # Navigation controls are generally near
                # the top of the page.
                if bounding_box["y"] <= 250:
                    return button

            except Exception:  # noqa: BLE001, S112
                continue

        except Exception:  # noqa: BLE001, S112
            continue

    return None


def collect_visible_navigation_links(
    page,
    homepage_url: str,
) -> list[tuple[str, str]]:
    """
    Collect all internal links currently present in the
    rendered DOM.
    """

    try:
        rendered_html = page.content()

        return collect_page_links(
            page_html=rendered_html,
            homepage_url=homepage_url,
        )

    except Exception:  # noqa: BLE001
        return []


def expand_navigation_sections(
    page,
    homepage_url: str,
) -> list[tuple[str, str]]:
    """
    Expand nested navigation sections and collect links
    revealed by them.

    This handles websites where the hamburger menu contains
    sections such as:

        Solutions >
        Platform >
        Resources >

    and their actual page links are hidden until clicked.
    """

    discovered_links = []

    navigation_controls = page.locator(
        "button, [role='button'], summary"
    )

    control_count = navigation_controls.count()

    for control_index in range(
        min(control_count, 60)
    ):

        try:
            control = navigation_controls.nth(
                control_index
            )

            if not control.is_visible(
                timeout=500
            ):
                continue

            control_text = (
                control.inner_text(
                    timeout=500
                )
                or ""
            ).strip().lower()

            aria_label = (
                control.get_attribute(
                    "aria-label"
                )
                or ""
            ).lower()

            combined_text = (
                f"{control_text} "
                f"{aria_label}"
            )

            # Never click the main hamburger/navigation
            # button again.
            if any(
                keyword in combined_text
                for keyword in MENU_KEYWORDS
            ):
                continue

            aria_expanded = (
                control.get_attribute(
                    "aria-expanded"
                )
                or ""
            ).lower()

            looks_like_expandable_section = (
                aria_expanded == "false"
                or any(
                    keyword in combined_text
                    for keyword
                    in EXPANDABLE_NAVIGATION_KEYWORDS
                )
            )

            if not looks_like_expandable_section:
                continue

            try:
                control.click(
                    timeout=1500
                )

            except Exception:  # noqa: BLE001, S112
                continue

            page.wait_for_timeout(
                MENU_WAIT
            )

            discovered_links.extend(
                collect_visible_navigation_links(
                    page=page,
                    homepage_url=homepage_url,
                )
            )

        except Exception:  # noqa: BLE001, S112
            continue

    return discovered_links


def discover_menu_links(
    company_domain: str,
) -> list[tuple[str, str]]:
    """
    Discover links exposed by dynamic navigation.

    A mobile-sized viewport is used because many modern
    websites switch to a hamburger menu at smaller widths.

    Process:

        Homepage
            ↓
        Open hamburger
            ↓
        Collect visible links
            ↓
        Expand nested sections
            ↓
        Collect newly revealed links
    """

    homepage_url = f"https://{company_domain}"

    discovered_menu_links = []

    with sync_playwright() as playwright:

        browser = launch_browser(playwright)

        page = browser.new_page(
            viewport={
                "width": 390,
                "height": 844,
            }
        )

        try:
            page.goto(
                homepage_url,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT,
            )

            page.wait_for_timeout(
                POST_LOAD_WAIT
            )

            menu_button = find_menu_button(
                page=page
            )

            if not menu_button:

                print(
                    "    No dynamic navigation "
                    "menu detected."
                )

                return []

            try:
                menu_button.click(
                    timeout=2000
                )

            except Exception as error:  # noqa: BLE001

                print(
                    f"    Could not open navigation menu: "
                    f"{error}"
                )

                return []

            page.wait_for_timeout(
                MENU_WAIT
            )

            print(
                "    Dynamic navigation menu opened."
            )

            # Collect links exposed immediately after
            # opening the menu.
            discovered_menu_links.extend(
                collect_visible_navigation_links(
                    page=page,
                    homepage_url=homepage_url,
                )
            )

            # Expand nested sections such as Solutions,
            # Platform, Resources, etc.
            discovered_menu_links.extend(
                expand_navigation_sections(
                    page=page,
                    homepage_url=homepage_url,
                )
            )

        except Exception as error:  # noqa: BLE001

            print(
                f"Menu discovery warning for "
                f"{homepage_url}: {error}"
            )

        finally:
            browser.close()

    # Remove duplicate URLs.
    unique_links = []
    seen_urls = set()

    for page_url, link_text in discovered_menu_links:

        if page_url in seen_urls:
            continue

        seen_urls.add(page_url)

        unique_links.append(
            (
                page_url,
                link_text,
            )
        )

    return unique_links


# ---------------------------------------------------------
# Page relevance scoring
# ---------------------------------------------------------


def score_page_relevance(
    page_url: str,
    link_text: str,
) -> int:
    """
    Calculate how valuable a page is for company
    intelligence extraction.

    The score prioritizes pages that can directly support
    the assignment's required fields.
    """

    normalized_url = page_url.lower()
    normalized_link_text = link_text.lower()

    page_score = 0

    for keyword, priority in PAGE_PRIORITY_KEYWORDS.items():

        keyword_in_url = (
            keyword in normalized_url
        )

        keyword_in_link_text = (
            keyword in normalized_link_text
        )

        if keyword_in_url:
            page_score = max(
                page_score,
                priority,
            )

        if keyword_in_link_text:
            page_score = max(
                page_score,
                priority,
            )

    # Give a small bonus when the URL itself clearly
    # identifies a relevant page.
    if normalized_url.count("/") >= 2:
        page_score += 1

    return page_score


def select_relevant_page_urls(
    discovered_links: list[tuple[str, str]],
) -> list[str]:
    """
    Rank discovered pages by assignment relevance and
    return only the highest-value pages.

    Duplicate URLs are merged.
    """

    page_scores = {}

    for page_url, link_text in discovered_links:

        relevance_score = score_page_relevance(
            page_url=page_url,
            link_text=link_text,
        )

        if relevance_score <= 0:
            continue

        existing_score = page_scores.get(
            page_url,
            0,
        )

        page_scores[page_url] = max(
            existing_score,
            relevance_score,
        )

    ranked_pages = sorted(
        page_scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return [
        page_url
        for page_url, _ in ranked_pages[
            :MAX_RELEVANT_PAGES
        ]
    ]


# ---------------------------------------------------------
# Main page discovery
# ---------------------------------------------------------


def discover_relevant_page_urls(
    page_html: str,
    homepage_url: str,
) -> list[str]:
    """
    Discover relevant internal company pages using:

    1. Normal homepage links.
    2. JavaScript-rendered hamburger navigation.
    3. Expandable navigation sections.

    Pages are then ranked according to their usefulness
    for company intelligence extraction.
    """

    company_domain = urlparse(
        homepage_url
    ).netloc

    # -----------------------------------------------------
    # Normal links from homepage HTML.
    # -----------------------------------------------------

    normal_links = collect_page_links(
        page_html=page_html,
        homepage_url=homepage_url,
    )

    # -----------------------------------------------------
    # Dynamic navigation links.
    # -----------------------------------------------------

    print(
        "  Checking dynamic navigation..."
    )

    menu_links = discover_menu_links(
        company_domain=company_domain,
    )

    # -----------------------------------------------------
    # Combine normal and dynamic links.
    # -----------------------------------------------------

    all_discovered_links = [
        *normal_links,
        *menu_links,
    ]

    relevant_page_urls = (
        select_relevant_page_urls(
            discovered_links=all_discovered_links,
        )
    )

    return relevant_page_urls


# ---------------------------------------------------------
# Page crawling
# ---------------------------------------------------------


def crawl_relevant_pages(
    company_domain: str,
    discovered_page_urls: list[str],
) -> dict[str, str]:
    """
    Crawl the homepage and all selected relevant pages.

    Each page is handled independently.

    Therefore:
        one failed page
            !=
        failed company

    This is important for resilience against:
    - timeouts
    - 404s
    - bot protection
    - empty pages
    - unexpected webpage errors
    """

    homepage_url = f"https://{company_domain}"

    crawled_pages = {}

    page_urls_to_crawl = [
        homepage_url,
        *discovered_page_urls,
    ]

    with sync_playwright() as playwright:

        browser = launch_browser(playwright)

        page = browser.new_page()

        for page_url in page_urls_to_crawl:

            print(
                f"  Crawling: {page_url}"
            )

            try:

                # -------------------------------------------------
                # Attempt normal navigation.
                # -------------------------------------------------

                try:
                    page.goto(
                        page_url,
                        wait_until="domcontentloaded",
                        timeout=PAGE_TIMEOUT,
                    )

                except Exception as error:  # noqa: BLE001

                    print(
                        f"    Navigation warning: "
                        f"{error}"
                    )

                    # -------------------------------------------------
                    # Recovery:
                    # If the browser received usable visible text
                    # before the timeout, keep it.
                    # -------------------------------------------------

                    try:
                        rendered_page_text = (
                            page.locator(
                                "body"
                            ).inner_text(
                                timeout=5000
                            )
                        )

                        if rendered_page_text.strip():

                            crawled_pages[
                                page_url
                            ] = rendered_page_text

                            print(
                                "    Usable page content "
                                "recovered after timeout."
                            )

                        else:

                            print(
                                "    No usable content "
                                "recovered."
                            )

                    except Exception:  # noqa: BLE001
                        print(
                            "    Could not recover "
                            "page content."
                        )

                    continue

                # -------------------------------------------------
                # Allow JavaScript-rendered content to settle.
                # -------------------------------------------------

                page.wait_for_timeout(
                    POST_LOAD_WAIT
                )

                # -------------------------------------------------
                # Extract visible rendered text.
                # -------------------------------------------------

                rendered_page_text = (
                    page.locator(
                        "body"
                    ).inner_text()
                )

                if rendered_page_text.strip():

                    crawled_pages[
                        page_url
                    ] = rendered_page_text

                    print(
                        f"    Success: "
                        f"{len(rendered_page_text):,} "
                        f"characters"
                    )

                else:

                    print(
                        "    Page loaded but "
                        "contained no visible text."
                    )

            except Exception as error:  # noqa: BLE001

                print(
                    f"    Failed to crawl "
                    f"{page_url}: {error}"
                )

        browser.close()

    return crawled_pages