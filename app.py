import os
import shutil

import streamlit as st

# Detect system Chromium when running on Streamlit Cloud
chromium_path = shutil.which("chromium")

if chromium_path:
    os.environ["CHROMIUM_PATH"] = chromium_path


from src.main import enrich_company

st.set_page_config(
    page_title="AI Lead Enrichment Agent",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 AI Lead Enrichment Agent")

st.write(
    "Enter a company domain to crawl its public website and "
    "generate structured company intelligence."
)


company_domain = st.text_input(
    "Company Domain",
    placeholder="e.g. stripe.com",
)


if st.button("🔍 Enrich Company", type="primary"):

    if not company_domain.strip():
        st.warning("Please enter a company domain.")
        st.stop()

    with st.spinner(
        "Crawling website and generating company intelligence..."
    ):
        try:
            company_lead = enrich_company(
                company_domain.strip()
            )

        except Exception as error:  # noqa: BLE001
            st.error(f"Unable to enrich company: {error}")
            st.stop()

    if company_lead.crawl_status != "success":
        st.warning(
            f"Company processing completed with status: "
            f"{company_lead.crawl_status}"
        )

    st.success("Company enrichment completed!")

    st.divider()

    # Company Overview
    st.subheader("🏢 Company Overview")
    st.write(company_lead.company_overview)

    # Target Audience
    st.subheader("🎯 Target Audience / ICP")
    st.write(company_lead.target_audience)

    # Contact Points
    st.subheader("📧 Contact Points")

    if company_lead.contact_points:
        for contact_point in company_lead.contact_points:
            st.write(f"- {contact_point}")
    else:
        st.write("No public contact points were discovered.")

    # Leadership
    st.subheader("👥 Leadership / Team")

    if company_lead.leadership_team:

        for member in company_lead.leadership_team:

            st.markdown(
                f"**{member.name}** — {member.role}"
            )

            if member.linkedin_url:
                st.markdown(
                    f"[LinkedIn Profile]({member.linkedin_url})"
                )

    else:
        st.write("No leadership information was discovered.")

    # Confidence
    st.subheader("📊 Data Confidence")

    st.progress(
        company_lead.confidence_score
    )

    st.write(
        f"{company_lead.confidence_score:.2f}"
    )

    # Pages
    with st.expander("🔎 Pages Analyzed"):

        for page_url in company_lead.pages_analyzed:
            st.write(page_url)

    # Warnings
    if company_lead.extraction_warnings:

        with st.expander("⚠️ Extraction Warnings"):

            for warning in company_lead.extraction_warnings:
                st.warning(warning)