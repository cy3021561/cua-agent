import os
from openai import OpenAI
from prompts.prompts import COMPUTER_USE_DOUBAO
from prompts.prompts import RESULT_CHECKING_WITH_IMAGES_PROMPT
from prompts.prompts import CODE_INTEGRATION_PROMPT
from typing import List
import glob
import pathlib
import json
import re
 


class AutomationState:
    def __init__(self, instruction: str, language: str = "English"):
        self.instruction = instruction
        self.language = language
        self.prev_image_b64 = None
        # store last two actions as list of dicts: {"thought": str, "action_str": str}
        self.actions = []

    def add_step(self, before_image_b64: str, thought: str, action_str: str):
        self.prev_image_b64 = before_image_b64
        self.actions.append({"thought": thought or "", "action_str": action_str or ""})
        if len(self.actions) > 2:
            self.actions = self.actions[-2:]


def build_messages_with_state(state: AutomationState, current_image_b64: str):
    messages = [
        {
            "role": "user",
            "content": COMPUTER_USE_DOUBAO.format(
                instruction=state.instruction,
                language=state.language
            )
        }
    ]
    # Optional previous image (first image)
    if state.prev_image_b64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Previous image:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{state.prev_image_b64}"
                }},
            ]
        })
    # Last two actions as text only (no images)
    for a in state.actions[-2:]:
        messages.append({
            "role": "assistant",
            "content": f"Thought: {a['thought']}\nAction: {a['action_str']}"
        })
    # Current image (second image) - already resized to screen size by caller
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Current image:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{current_image_b64}"
            }},
        ]
    })
    return messages


def call_ui_grounding_model_with_messages(messages) -> str:
    client = OpenAI(
        base_url=os.environ["TGI_BASE_URL"],
        api_key=os.environ["HF_TOKEN"]
    )
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    return chat_completion.choices[0].message.content


def call_ui_grounding_model(base64_image: str, instruction: str, language: str = "English") -> str:
    """
    Make a model call with a base64 image and instruction for UI grounding.
    
    Args:
        base64_image: Base64 encoded image string (without data:image/png;base64, prefix)
        instruction: The instruction text to send to the model
        language: Language for the prompt (default: "English")
    
    Returns:
        str: The complete model response as a string
    """
    # Initialize huggingface compatible OpenAI client
    client = OpenAI(
        base_url=os.environ["TGI_BASE_URL"],
        api_key=os.environ["HF_TOKEN"]
    )
    
    # Prepare messages with instruction and image
    messages = [
        {
            "role": "user", 
            "content": COMPUTER_USE_DOUBAO.format(instruction=instruction, language=language)
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    # Make the non-streaming API call
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        top_p=None,
        temperature=0.0,
        max_tokens=400,
        stream=False,
        seed=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None
    )

    # Get the complete response
    raw_response = chat_completion.choices[0].message.content

    return raw_response

def call_result_checking_model(task_description: str, expected_view_base64: str, current_view_base64: str) -> dict:
    """
    Determine if a task is finished by comparing an expected end-state image and the current image.

    Args:
        task_description: The description of the task to be checked.
        expected_view_base64: Base64 of the expected end-state screenshot (PNG).
        current_view_base64: Base64 of the current screenshot (PNG).

    Returns:
        dict: {"thoughts": str, "result": bool}
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    instruction = RESULT_CHECKING_WITH_IMAGES_PROMPT.format(task_description=task_description)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Expected end view:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{expected_view_base64}"}},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current view:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_view_base64}"}},
            ]
        }
    ]

    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    raw_response = chat_completion.choices[0].message.content or ""

    thoughts_match = re.search(r"Thought:\s*(.*?)(?:\n\s*Action:|\Z)", raw_response, flags=re.DOTALL | re.IGNORECASE)
    thoughts = thoughts_match.group(1).strip() if thoughts_match else raw_response.strip()

    result_match = re.search(r"finished\(content=['\"]?(true|false)['\"]?\)", raw_response, flags=re.IGNORECASE)
    result_str = result_match.group(1).lower() if result_match else "false"
    result_bool = True if result_str == "true" else False

    return {"thoughts": thoughts, "result": result_bool}


def _load_automation_step_snippets(snippets_dir: str = "./data/automation_code") -> List[str]:
    """
    Load all Python code snippets from the automation steps directory, sorted by filename.

    Args:
        snippets_dir: Directory that contains step files like automation_step_*.py

    Returns:
        List[str]: List of code snippet strings in filename order
    """
    # Resolve absolute path while allowing default relative usage
    dir_path = pathlib.Path(snippets_dir).resolve()
    pattern = str(dir_path / "automation_step_*.py")
    files = sorted(glob.glob(pattern))
    snippets: List[str] = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                snippets.append(f.read())
        except Exception:
            # Skip unreadable files; we want best-effort aggregation
            continue
    return snippets


def call_code_integration_model_with_snippets(snippets: List[str]) -> str:
    """
    Call gpt-4o with CODE_INTEGRATION_PROMPT and provided snippets to generate a unified PyAutoGUI script.

    Args:
        snippets: List of code snippet strings composing the task steps.

    Returns:
        str: The model's raw response, expected to be a complete PyAutoGUI script.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # Join snippets with clear boundaries to help the model
    joined_snippets = "\n\n# ===== SNIPPET SEPARATOR =====\n\n".join(snippets)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": CODE_INTEGRATION_PROMPT.format(code_snippets=joined_snippets)
                }
            ]
        }
    ]

    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    raw_response = chat_completion.choices[0].message.content
    return raw_response


def call_code_integration_model_from_dir(snippets_dir: str = "./data/automation_code") -> str:
    """
    Convenience wrapper that loads snippets from a directory and calls the integration model.

    Args:
        snippets_dir: Directory containing automation step files.

    Returns:
        str: The model's raw response, expected to be a complete PyAutoGUI script.
    """
    snippets = _load_automation_step_snippets(snippets_dir)
    return call_code_integration_model_with_snippets(snippets)