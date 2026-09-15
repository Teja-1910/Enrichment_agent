# AI Lead Enrichment Agent

## Overview
Python-based autonomous agent that crawls public company websites,
extracts relevant content, and uses an LLM to generate structured
company intelligence.

## Architecture
Company Domain
      ↓
Playwright Browser
      ↓
Relevant Page Discovery
      ↓
Rendered Text Extraction
      ↓
Content Cleaning
      ↓
LLM Structured Extraction
      ↓
Pydantic Validation
      ↓
output/output.json

## Technologies
- Python
- Playwright
- BeautifulSoup
- Groq LLM API
- Pydantic
- python-dotenv

## Setup

Create .env:

GROQ_API_KEY=your_api_key_here

Install dependencies:

pip install -r requirements.txt

Install browser:

playwright install chromium

Run:

python src/main.py

## Target Companies

- postman.com
- supabase.com
- vapi.ai

## Output

Generated intelligence is available in:

output/output.json

## Notes

The crawler handles JavaScript-rendered websites, relevant-page
discovery, page failures, timeouts, and missing content without
stopping the entire pipeline.