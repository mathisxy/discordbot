from os import getenv

from dotenv import load_dotenv

load_dotenv()


def require_env(key: str):

    value = getenv(key)

    if not value:
        raise Exception(f"{key} is required")

    return value


class Config:

    DISCORD_TOKEN: str = require_env("DISCORD_TOKEN")
    MISTRAL_API_KEY: str = require_env("MISTRAL_API_KEY")
    CLAUDE_API_KEY: str = require_env("CLAUDE_API_KEY")
    GEMINI_API_KEY: str = require_env("GEMINI_API_KEY")
    OPENAI_API_KEY: str = require_env("OPENAI_API_KEY")
    AZURE_API_KEY: str = require_env("AZURE_API_KEY")
    AZURE_BASE_URL: str = require_env("AZURE_BASE_URL")