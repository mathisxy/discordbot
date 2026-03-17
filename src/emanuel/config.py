from os import getenv

from dotenv import load_dotenv

load_dotenv()


def require_env(key: str):

    value = getenv(key)

    if not value:
        raise Exception(f"{key} is required")

    return value

def getenv_csv(key: str) -> list[str]:
    value = getenv(key, "")
    items = value.split(",")
    return [item.strip() for item in items if item.strip() != ""]


class Config:

    DISCORD_TOKEN: str = require_env("DISCORD_TOKEN")

    LLM_PROVIDER: str = require_env("LLM_PROVIDER")
    LLM_STREAMING: str = require_env("LLM_STREAMING")
    MCP_SERVER_URLS: list[str] = getenv_csv("MCP_SERVER_URLS")

    MISTRAL_API_KEY: str = getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL: str = getenv("MISTRAL_MODEL", "")

    CLAUDE_API_KEY: str = getenv("CLAUDE_API_KEY", "")
    CLAUDE_MODEL: str = getenv("CLAUDE_MODEL", "")

    GEMINI_API_KEY: str = getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = getenv("GEMINI_MODEL", "")

    OPENAI_API_KEY: str = getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = getenv("OPENAI_MODEL", "")

    AZURE_API_KEY: str = getenv("AZURE_API_KEY", "")
    AZURE_BASE_URL: str = getenv("AZURE_BASE_URL", "")
    AZURE_MODEL: str = getenv("AZURE_MODEL", "")

    OLLAMA_MODEL: str = getenv("OLLAMA_MODEL", "")
    OLLAMA_KEEP_ALIVE: str = getenv("OLLAMA_KEEP_ALIVE", "")

    CHAT_SYSTEM_MESSAGE: str = getenv("CHAT_SYSTEM_MESSAGE", "")
    VOICE_SYSTEM_MESSAGE: str = getenv("VOICE_SYSTEM_MESSAGE", "")

    MAX_TURNS: int = int(require_env("MAX_TURNS"))