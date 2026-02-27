import os
from dotenv import load_dotenv

_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEAR_API_KEY = os.getenv("BEAR_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
BEAR_MODEL = "bear-1.2"

BEAR_API_URL = "https://api.thetokencompany.com/v1/compress"

AGGRESSIVENESS_LEVELS = [0.05, 0.1, 0.3, 0.4, 0.5, 0.7]

DATASET_NAME = "openai/mrcr"
N_NEEDLES = 8  # Filter dataset to 8-needle variant

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

MAX_COMPLETION_TOKENS = 4096  # MRCR answers are long creative writing

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SYSTEM_PROMPT = (
    "Read the conversation and follow the final instruction. "
    "Reproduce the requested content exactly, beginning your response with the specified string."
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
