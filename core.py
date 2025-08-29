import os
from openai import OpenAI
from prompts.prompts import COMPUTER_USE_DOUBAO, RESULT_CHECKING_PROMPT
 


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

# TODO: Should we add a expected result description to the prompt?
def call_result_checking_model(task_description: str, expected_result_description: str, current_screenshot_base64: str) -> bool:
    """
    Call a vlm model(e.g. gpt-4o sereis or still using UI-TARS model) with current screenshot to check if the task in task_description is finished.

    Args:
        task_description: The description of the task to be checked
        current_screenshot_base64: The base64 encoded current screenshot

    Returns:
        bool: True if the task is finished, False otherwise
    """
    print(f"Calling result checking model with task description: {task_description}")
    print(f"Expected result description: {expected_result_description}")
    # client = OpenAI(
    #     api_key=os.environ["OPENAI_API_KEY"]
    # )
    client = OpenAI(
        base_url=os.environ["TGI_BASE_URL"],
        api_key=os.environ["HF_TOKEN"]
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": RESULT_CHECKING_PROMPT.format(task_description=task_description, expected_result_description=expected_result_description)
                }
            ]
        }
    ]

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Before image:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{current_screenshot_base64}"
                    }
                }
            ]
        }
    )

    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    # Get the complete response
    raw_response = chat_completion.choices[0].message.content
    # print(f"Raw response: {raw_response}")

    return raw_response