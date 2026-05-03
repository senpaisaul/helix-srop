import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env BEFORE Settings() instantiates. Otherwise pydantic-settings prefers
# whatever is already in os.environ over the file — which silently breaks dev
# when a parent shell ships an empty ANTHROPIC_API_KEY (Claude Code does this).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"
    chroma_persist_dir: str = "./chroma_db"

    google_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    adk_model: str = "gemini-2.0-flash"

    llm_timeout_seconds: int = 30
    tool_timeout_seconds: int = 10


settings = Settings()

# google-adk's runtime instantiates google.genai.Client(), which reads the
# API key from os.environ — not from our Settings object. Mirror across
# every provider key we care about so .env-only setups work everywhere.
for _name in ("google_api_key", "anthropic_api_key", "openai_api_key"):
    _val = getattr(settings, _name, "")
    _env_name = _name.upper()
    if _val and not os.environ.get(_env_name):
        os.environ[_env_name] = _val
