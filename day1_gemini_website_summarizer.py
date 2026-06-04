#!/usr/bin/env python3
"""
LLM Engineering - Week 1 Day 1 (Gemini variant)

Adapted from day1.ipynb to use Google Gemini instead of OpenAI:
- GOOGLE_API_KEY in .env (starts with AIza)
- OpenAI Python SDK pointed at Gemini's OpenAI-compatible endpoint
- Model: gemini-2.5-flash-lite

Requires: pip install openai python-dotenv requests beautifulsoup4

Usage:
  python day1_gemini_website_summarizer.py --demo
  python day1_gemini_website_summarizer.py --url https://edwarddonner.com
  python day1_gemini_website_summarizer.py --exercise
"""

from __future__ import annotations

import argparse
import os
import sys

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# --- Config (your notebook changes) ---
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash-lite"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

SYSTEM_PROMPT = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

USER_PROMPT_PREFIX = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


# --- Scraper (from week1/scraper.py) ---
def fetch_website_contents(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:2_000]


# --- Gemini client (your notebook changes) ---
def create_gemini_client() -> OpenAI:
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: GOOGLE_API_KEY not found in environment.\n"
            "Add to .env: GOOGLE_API_KEY=AIza...\n"
            "Get a key: https://aistudio.google.com/",
            file=sys.stderr,
        )
        sys.exit(1)
    if not api_key.startswith("AIza"):
        print(
            "WARNING: Key does not start with AIza — check you copied the Google key.",
            file=sys.stderr,
        )
    return OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)


def messages_for(website: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_PREFIX + website},
    ]


def chat(gemini: OpenAI, messages: list[dict[str, str]]) -> str:
    response = gemini.chat.completions.create(
        model=GEMINI_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def summarize_url(gemini: OpenAI, url: str) -> str:
    website = fetch_website_contents(url)
    return chat(gemini, messages_for(website))


# --- Notebook demos ---
def run_demo(gemini: OpenAI) -> None:
    print("=== Quick math test ===")
    r = chat(
        gemini,
        [{"role": "user", "content": "what is 12+2?"}],
    )
    print(r)

    print("\n=== Hello message ===")
    r = chat(
        gemini,
        [{"role": "user", "content": "Hello! This is my first message to you. Hi!"}],
    )
    print(r)

    print("\n=== 2 + 2 with system prompt ===")
    r = chat(
        gemini,
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
    )
    print(r)


def run_exercise(gemini: OpenAI) -> None:
    """Your custom Step 1–4 exercise from the notebook."""
    system_prompt = "you're a system design expert, who knows complete SDLC"
    user_prompt = "give me good suggestion on how to begin with system design"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    print("=== System design exercise ===")
    print(chat(gemini, messages))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 1 Day 1 website summarizer using Google Gemini"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick API smoke tests from the notebook",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Summarize a single URL (e.g. https://edwarddonner.com)",
    )
    parser.add_argument(
        "--exercise",
        action="store_true",
        help="Run the system-design prompt exercise",
    )
    args = parser.parse_args()

    gemini = create_gemini_client()

    if args.demo:
        run_demo(gemini)

    if args.url:
        print(f"\n=== Summary: {args.url} ===\n")
        print(summarize_url(gemini, args.url))

    if args.exercise:
        run_exercise(gemini)

    if not (args.demo or args.url or args.exercise):
        parser.print_help()
        print("\nExample: python day1_gemini_website_summarizer.py --url https://edwarddonner.com")


if __name__ == "__main__":
    main()
