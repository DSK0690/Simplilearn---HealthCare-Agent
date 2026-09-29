"""Central configuration."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    def load_dotenv(*a, **k):
        return False

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = Path(os.getenv("HEALTHCARE_DB_PATH", DATA_DIR / "healthcare.db"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
MODEL_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SEARCH_ONLINE = os.getenv("SEARCH_ONLINE", "true").lower() in ("1", "true", "yes")
EMBED_DIM = 384

DISCLAIMER = (
    "This assistant supports administrative and informational tasks only. "
    "It does not provide a diagnosis or replace advice from a qualified clinician."
)
