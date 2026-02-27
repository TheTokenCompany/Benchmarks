import os
from dotenv import load_dotenv

_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEAR_API_KEY = os.getenv("BEAR_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
BEAR_MODEL = "bear-1.2"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.2")

BEAR_API_URL = "https://api.thetokencompany.com/v1/compress"

AGGRESSIVENESS_LEVELS = [0.05, 0.1, 0.3, 0.4, 0.5, 0.7]

DATASET_NAME = "stanfordnlp/coqa"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SYSTEM_PROMPT = (
    "Answer the question based only on the provided story and conversation history. "
    "Keep your answer concise — extract the relevant phrase or sentence from the story."
)

CONFIGS = {
    "control": {"compressed": False, "aggressiveness": None},
    "bear_0.05": {"compressed": True, "aggressiveness": 0.05},
    "bear_0.1": {"compressed": True, "aggressiveness": 0.1},
    "bear_0.3": {"compressed": True, "aggressiveness": 0.3},
    "bear_0.4": {"compressed": True, "aggressiveness": 0.4},
    "bear_0.5": {"compressed": True, "aggressiveness": 0.5},
    "bear_0.7": {"compressed": True, "aggressiveness": 0.7},
}
