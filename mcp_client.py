import os
import shutil
import sys
import asyncio
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Support both environment-variable names.
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
UVX_COMMAND = shutil.which("uvx") or "uvx"


def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""

    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )

    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """
    Preserve the current Windows/Conda environment and add MCP API keys.
    """

    env = os.environ.copy()

    for key, value in updates.items():
        if value:
            env[key] = value

    return env


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
)


# =========================================================
# MCP client
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },

        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "aviationstack-mcp",
            ],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },

        "weather": {
            "transport": "stdio",

            # Uses the Python executable from the active Conda environment.
            "command": sys.executable,

            # Uses the weather server inside the current project folder.
            "args": [
                str(WEATHER_SERVER_PATH),
            ],

            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """
    Load one tool from one MCP server.

    This prevents a broken weather or AviationStack server from
    crashing an unrelated Tavily request.
    """

    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )

    elif server_name == "aviationstack":
        _require_env(
            "AVIATION_STACK_API_KEY",
            AVIATION_STACK_API_KEY,
        )

        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv, reopen the terminal, "
                "activate the travel environment, and run "
                "`uvx --version`."
            )

    elif server_name == "weather":
        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(
                f"Weather MCP server not found: "
                f"{WEATHER_SERVER_PATH}"
            )

    # Important: load only the requested MCP server.
    tools = await client.get_tools(
        server_name=server_name,
    )

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(item.name for item in tools)
            )
            or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


# =========================================================
# MCP connection test
# =========================================================

async def get_all_tools() -> None:
    """
    Test every MCP server independently.

    One failed server will not stop the remaining tests.
    """

    for server_name in (
        "tavily",
        "aviationstack",
        "weather",
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name,
            )

            tool_names = (
                ", ".join(
                    tool.name
                    for tool in tools
                )
                or "no tools"
            )

            print(
                f"{server_name}: OK -> {tool_names}"
            )

        except Exception as exc:
            print(
                f"{server_name}: FAILED -> "
                f"{type(exc).__name__}: {exc}"
            )


# =========================================================
# Tavily MCP
# =========================================================

async def tavily_mcp_search(query: str):
    if not TAVILY_API_KEY:
        print("TAVILY_API_KEY is missing. Falling back to DuckDuckGo search.")
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            result = await asyncio.to_thread(search.invoke, query)
            return result
        except Exception as exc:
            print(f"DuckDuckGo search fallback failed: {exc}")
            return f"Search failed. No search results available for: {query}"

    try:
        search_tool = await _get_server_tool(
            "tavily",
            "tavily_search",
        )
        return await search_tool.ainvoke(
            {
                "query": query,
            }
        )
    except Exception as exc:
        print(f"Tavily MCP search failed ({exc}). Falling back to DuckDuckGo search.")
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            result = await asyncio.to_thread(search.invoke, query)
            return result
        except Exception as e:
            return f"Search failed. Error: {exc}. DDG Error: {e}"


# =========================================================
# AviationStack MCP
# =========================================================

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    def get_mock_data():
        if tool_name == "list_airports":
            return [
                {"airport_name": "John F. Kennedy International", "iata_code": "JFK", "city_name": "New York", "country_name": "United States"},
                {"airport_name": "Heathrow Airport", "iata_code": "LHR", "city_name": "London", "country_name": "United Kingdom"},
                {"airport_name": "Haneda Airport", "iata_code": "HND", "city_name": "Tokyo", "country_name": "Japan"},
                {"airport_name": "Dubai International", "iata_code": "DXB", "city_name": "Dubai", "country_name": "United Arab Emirates"},
                {"airport_name": "Singapore Changi", "iata_code": "SIN", "city_name": "Singapore", "country_name": "Singapore"},
                {"airport_name": "Charles de Gaulle", "iata_code": "CDG", "city_name": "Paris", "country_name": "France"}
            ]
        elif tool_name == "list_airlines":
            return [
                {"airline_name": "Delta Air Lines", "iata_code": "DL"},
                {"airline_name": "Emirates", "iata_code": "EK"},
                {"airline_name": "British Airways", "iata_code": "BA"},
                {"airline_name": "All Nippon Airways", "iata_code": "NH"},
                {"airline_name": "Singapore Airlines", "iata_code": "SQ"},
                {"airline_name": "Air France", "iata_code": "AF"}
            ]
        return []

    if not AVIATION_STACK_API_KEY:
        print("AVIATION_STACK_API_KEY is missing. Returning mock aviation data.")
        return get_mock_data()

    try:
        aviation_tool = await _get_server_tool(
            "aviationstack",
            tool_name,
        )
        return await aviation_tool.ainvoke(
            tool_args or {}
        )
    except Exception as exc:
        print(f"AviationStack MCP call failed ({exc}). Returning mock aviation data.")
        return get_mock_data()


# =========================================================
# Weather MCP
# =========================================================

async def weather_mcp_search(city: str):
    weather_tool = await _get_server_tool(
        "weather",
        "get_current_weather",
    )

    return await weather_tool.ainvoke(
        {
            "city": city,
        }
    )


async def forecast_mcp_search(city: str):
    forecast_tool = await _get_server_tool(
        "weather",
        "get_forecast",
    )

    return await forecast_tool.ainvoke(
        {
            "city": city,
        }
    )


# =========================================================
# Destination extractor
# =========================================================

def extract_destination(query: str) -> str:
    prompt = f"""
Extract only the destination city or country from the travel request.

Travel request:
{query}

Return only the destination name.
Do not add any explanation.
"""

    response = llm.invoke(prompt)

    destination = str(
        response.content
    ).strip()

    if not destination:
        raise ValueError(
            "The destination could not be extracted."
        )

    return destination