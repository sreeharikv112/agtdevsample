import os
import logging
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_TOKENS = 300
TEMPERATURE = 0.7
TASK_FILE_PATH = "tasks.txt"

# ---------------------------
# Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

HF_API_TOKEN = os.getenv("HUGGINGFACEKEY")
if not HF_API_TOKEN:
    raise ValueError("HUGGINGFACEKEY not found in environment variables.")

hf_client = InferenceClient(token=HF_API_TOKEN)


# ---------------------------
# File Handling
# ---------------------------
def read_task_file(file_path: str) -> Optional[str]:
    """Read tasks from a file."""
    if not os.path.exists(file_path):
        logger.error("File does not exist: %s", file_path)
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        logger.exception("Failed to read file: %s", file_path)
        return None


# ---------------------------
# Prompt Builder
# ---------------------------
def build_prompt(tasks: str) -> str:
    """Construct prompt for task prioritization."""
    return f"""
You are a helpful assistant. Categorize the following tasks into 3 priority levels:

- High Priority
- Medium Priority
- Low Priority

Tasks:
{tasks}

Return output strictly in this format:

High Priority:
- Task 1
- Task 2

Medium Priority:
- Task 3
- Task 4

Low Priority:
- Task 5
- Task 6
""".strip()


# ---------------------------
# AI Processing
# ---------------------------
def generate_task_summary(tasks: str) -> Optional[str]:
    """Generate prioritized task summary using Hugging Face model."""
    prompt = build_prompt(tasks)

    try:
        response = hf_client.chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        return response.choices[0].message["content"]

    except Exception as e:
        logger.exception("Error during model inference")
        return None


# ---------------------------
# Main Execution
# ---------------------------
def main() -> None:
    tasks = read_task_file(TASK_FILE_PATH)

    if not tasks:
        logger.warning("No tasks found. Exiting.")
        return

    summary = generate_task_summary(tasks)

    if summary:
        print("\nTask Summary:\n")
        print(summary)
    else:
        logger.error("Failed to generate task summary.")


if __name__ == "__main__":
    main()