"""AI-powered financial insights agent (Groq + Phidata + yfinance tools).

Usable as a library (`get_insights`) or run directly for a CLI demo.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools

load_dotenv()

COMMON_SYMBOLS = {
    "infosys": "INFY",
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
}


def get_company_symbol(company: str) -> str:
    """Resolve a company name or ticker to a ticker symbol.

    Args:
        company (str): A company name (e.g. "Tesla") or ticker (e.g. "TSLA").

    Returns:
        str: The best-guess ticker symbol for the company.
    """
    key = company.strip().lower()
    return COMMON_SYMBOLS.get(key, company.strip().upper())


def build_agent() -> Agent:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
        )
    return Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        tools=[
            YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
            get_company_symbol,
        ],
        instructions=[
            "Use tables to display data.",
            "If you need to find the symbol for a company, use the get_company_symbol tool.",
            "Be concise and highlight the most decision-relevant numbers.",
        ],
        markdown=True,
    )


def get_insights(symbol: str, question: str | None = None) -> str:
    """Ask the AI finance agent about `symbol` (or any custom `question`).

    Returns the agent's markdown-formatted response as a string.
    """
    agent = build_agent()
    prompt = question or (
        f"Summarize the latest fundamentals and analyst recommendations for {symbol}. Show in tables."
    )
    response = agent.run(prompt)
    return response.content


if __name__ == "__main__":
    args = sys.argv[1:]
    prompt = " ".join(args) if args else (
        "Summarize and compare analyst recommendations and fundamentals for TSLA and MSFT. Show in tables."
    )
    build_agent().print_response(prompt, stream=True)
