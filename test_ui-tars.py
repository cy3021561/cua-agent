# pip install openai
import asyncio
import io
import os
import re
import json
import base64
import time
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from computer import Computer

from action_parser import add_box_token, parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize, parse_action, convert_point_to_coordinates
from utils import visualize_actions_on_image, execute_pyautogui_code, get_screenshot_base64, get_size_from_base64
from core import call_ui_grounding_model, call_result_checking_model, AutomationState, build_messages_with_state, call_ui_grounding_model_with_messages, call_code_integration_model_from_dir

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or ""
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY") or ""
os.environ["TGI_BASE_URL"] = os.getenv("TGI_BASE_URL") or ""

MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT = 3360, 2100
RESIZED_MODEL_IMG_WIDTH, RESIZED_MODEL_IMG_HEIGHT = 2880, 1800
SCREEN_WIDTH, SCREEN_HEIGHT = 1680, 1050
FACTOR = 28


def run_images_testing():
    """
    Test function that runs through a series of test images and instructions to validate
    the UI automation pipeline.
    """
    test_instructions = [
                        "I need to click on the 'Member ID' input field.",\
                        "Member ID input field is selected, I need to type in 'E01247555' into the Member ID filed.",\
                        "I need to click on the Month input field.",\
                        "Month input field is selected, I need to type in '11' into the Month input field.",\
                        "Day input field is selected, I need to type in '01' into the Day input field.",\
                        "Year input field is selected, I need to type in '1992' into the Year input field.",\
                        "I need to click on the 'Search' button.",\
                        "I need to click on 'Benefit Details'.",\
                        "I need to click on the dropdown date range.",\
                        "I need to click on the 'save' button.",\
                        "I need to close this tab."
                        ]

    for i, cur_test_instruction in enumerate(test_instructions):
        print(f'=== Test image {i+1} ===')
        cur_test_img_path = f'./data/test_images/test_img_{i+1}.png'
       
        # Read image and transform it to base64
        image = Image.open(cur_test_img_path)

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_width, image_height = image.size
        
        # Call the model with the image and instruction
        raw_response = call_ui_grounding_model(
            base64_image=img_str,
            instruction=cur_test_instruction,
            language="English"
        )

        print("=== RAW Response ===")
        print(raw_response)
        # raw_response = "Thought: I noticed that the input box for the Member ID is already selected, with a blue border around it. This is exactly the state I was looking for, so I can go ahead and enter the ID number E01247555 directly. There's no need for any additional clicks or actions; I can just start typing on the keyboard.\nAction: type(content='E0147555')"

        # raw_response = "Thought: I noticed that there is a search box on the page labeled \"Member ID,\" which is exactly what I need to click on. The input field is located in the upper half of the page, and I need to move my mouse over it and click to activate it so that I can enter the member ID information.\nAction: click(start_box='(1103,814)')"

        structured_actions = parse_action_to_structure_output(
            raw_response,
            factor=FACTOR,
            origin_resized_height=MODEL_IMG_HEIGHT,
            origin_resized_width=MODEL_IMG_WIDTH
        )
        print("=== Generated Structured Actions ===")
        print(structured_actions)
        
        # Debug: Print action details
        for idx, action in enumerate(structured_actions):
            print(f"Action {idx + 1}:")
            print(f"  Type: {action['action_type']}")
            print(f"  Inputs: {action['action_inputs']}")
            print(f"  Thought: {action['thought']}")
            print("---")

        # Visualize actions on the image and save it
        output_path = f'./data/test_images/coordinate_process_image_{i+1}.png'
        visualize_actions_on_image(
            image=image,
            structured_actions=structured_actions,
            output_path=output_path,
            title=cur_test_instruction
        )

        # Generate and execute PyAutoGUI code for computer automation
        pyautogui_code = parsing_response_to_pyautogui_code(
            structured_actions,
            image_height=SCREEN_HEIGHT,
            image_width=SCREEN_WIDTH
        )

        print("=== Generated PyAutoGUI code ===")
        print(pyautogui_code)

        # Execute the generated code (uncomment to actually execute)
        # success, result = execute_pyautogui_code(pyautogui_code)
        # if success:
        #     print("PyAutoGUI code executed successfully")
        # else:
        #     print(f"Failed to execute PyAutoGUI code: {result}")

        # break

def demo_local_cua_step_automation(instruction: str, step_idx: int, max_iterations: int = 5):
    """
    Demo function showing continuous automation with short history:
    - At most two images per model call (previous + current)
    - Last two actions as text for reasoning
    - Early stop when model emits finished
    """
    print(f"=== Step {step_idx} Started ===")
    print(f"Instruction: {instruction}")
    print(f"Max iterations: {max_iterations}")

    state = AutomationState(instruction=instruction, language="English")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        try:
            # 1) Capture current screenshot (will be the second image)
            print("Capturing screenshot...")
            cur_b64 = get_screenshot_base64(width=RESIZED_MODEL_IMG_WIDTH, height=RESIZED_MODEL_IMG_HEIGHT)
            print("Screenshot captured and converted to base64")

            # 2) Build messages (<=2 images) and call model
            messages = build_messages_with_state(state, cur_b64)
            print("Calling UI grounding model with history...")
            raw_response = call_ui_grounding_model_with_messages(messages)

            # 3) Parse the response into actions
            structured_actions = parse_action_to_structure_output(
                raw_response,
                factor=FACTOR,
                origin_resized_height=RESIZED_MODEL_IMG_HEIGHT,
                origin_resized_width=RESIZED_MODEL_IMG_WIDTH
            )

            # 4) Early stop if finished
            if any(a.get("action_type") == "finished" for a in structured_actions):
                print("Task completed (model emitted finished).")
                break

            # 5) Generate PyAutoGUI code
            pyautogui_code = parsing_response_to_pyautogui_code(
                structured_actions,
                image_height=SCREEN_HEIGHT,
                image_width=SCREEN_WIDTH
            )
            # print("--------------------------------")
            # print("Generated PyAutoGUI code")
            # print(pyautogui_code)
            # print("--------------------------------")

            # Save every pyautogui code to a file for debugging
            os.makedirs("./data/automation_code", exist_ok=True)
            with open(f"./data/automation_code/automation_step_{step_idx}_{iteration + 1}.py", "w") as f:
                f.write(pyautogui_code)

            # 6) Execute code unless parser signaled DONE
            if pyautogui_code == "DONE":
                print("Task completed (parser signals DONE).")
                break

            print("Executing PyAutoGUI code...")
            success, result = execute_pyautogui_code(pyautogui_code)
            if success:
                print("Actions executed successfully")
            else:
                print(f"Failed to execute actions: {result}")
                # Choose to continue or break based on policy
                # break

            # 7) Visualize the actions using current image
            output_path = f"./data/screenshots/automation_step_{step_idx}_{iteration + 1}.png"
            visualize_actions_on_image(
                image=cur_b64,
                structured_actions=structured_actions,
                output_path=output_path,
                title=f"Automation Step {iteration + 1}: {instruction[:30]}..."
            )
            print(f"Actions visualized and saved to: {output_path}")

            # 8) Save step memory (previous image + last action summary)
            thought = structured_actions[0].get("thought", "") if structured_actions else ""
            first = structured_actions[0] if structured_actions else {"action_type": "", "action_inputs": {}}
            action_str = f"{first['action_type']}(" + ", ".join(
                f"{k}='{v}'" for k, v in first.get("action_inputs", {}).items()
            ) + ")"
            state.add_step(before_image_b64=cur_b64, thought=thought, action_str=action_str)

            # Short settle time
            time.sleep(1.0)

        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {str(e)}")
            break

    print("\n=== Step Ended ===")

def demo_local_result_checking():
    """
    Demo function showing result checking: screenshot -> model -> result checking
    """
    test_instructions = [
        "I need to click on the 'Member ID' input field.",\
        "Member ID input field is selected, I can type in 'E01257444' into the Member ID filed directly without clicking.",\
        "I need to click on the Month input field.",\
        "Month input field is selected, I can type in '11' into the Month input field directly without clicking.",\
        "I need to click on the Day input field.",\
        "Day input field is selected, I can type in '01' into the Day input field directly without clicking.",\
        "I need to click on the Year input field.",\
        "Year input field is selected, I can type in '1992' into the Year input field directly without clicking.",\
        "I need to click on the 'Search' button.",\
        "I need to click on 'Benefit Details'.",\
        "I need to click on the dropdown date range to open the file save dialog.",\
        "I need to click on the 'save' button.",\
        ]
    
    expected_result_descriptions = [
        "The page should show that the Member ID input field is selected/focused.",
        "The Member ID input field should have the text 'E01257444'.",
        "The page should show that the Month(MM) input field is selected/focused.",
        "The Month input field should have the text '11'.",
        "The page should show that the Day(DD) input field is selected/focused.",
        "The Day input field should have the text '01'.",
        "The page should show that the Year(YYYY) input field is selected/focused.",
        "The Year input field should have the text '1992'.",
        "The page should log in successfully, and the page should show the patient's insurance information.",
        "There will be a dropdown date range on the page below the benefit details.",
        "A file save dialog has popped up on the screen with save button.",
        "The page should show that the saved file is downloaded.",
        ]
    
    for i, cur_test_instruction in enumerate(test_instructions):
        step_idx = i + 1
        img_idx = step_idx + 1
        print(f'\n=== Step {step_idx} result checking: image {img_idx} ===')
        finished_test_img_path = f'./data/test_images/test_img_{img_idx}.png'

        # Read images and convert to base64
        finished_image = Image.open(finished_test_img_path)
        buffered = BytesIO()
        finished_image.save(buffered, format="PNG")
        finished_img_str = base64.b64encode(buffered.getvalue()).decode()

        # Call the result checking model
        result = call_result_checking_model(cur_test_instruction, expected_result_descriptions[i], finished_img_str)
        print(f"{result}")

async def run_docker_container(image_name: str):
    # 0) Run the docker container and get the screen size and screenshot size
    computer = Computer(
        os_type="linux",
        provider_type="docker",
        image=image_name,
        name="my-cua-container",
        noVNC_port=6901,
        port=8000
    )

    await computer.run()

    print("VNC Access: http://localhost:6901")
    print("API: http://localhost:8000")

    # 1) Get the screenshot size and screen size
    screenshot_bytes = await computer.interface.screenshot()
    screenshot_size = get_size_from_base64(screenshot_bytes)
    image_width, image_height = screenshot_size[0], screenshot_size[1]
    
    # 2) Get the screen size for pyautogui execution
    screen_size = await computer.interface.get_screen_size()
    screen_width, screen_height = screen_size['width'], screen_size['height']

    # 3) Open the browser
    _ = await computer.interface.run_command(
        "bash -lc 'nohup xdg-open https://www.brmsprovidergateway.com/provideronline/search.aspx "
        ">/dev/null 2>&1 </dev/null &'"
    )
    time.sleep(5)

    return computer, image_width, image_height, screen_width, screen_height

async def demo_docker_cua_step_automation(instruction: str, computer: Computer, image_width: int, image_height: int, screen_width: int, screen_height: int, step_idx: int, max_iterations: int = 5):
    """
    Demo function showing continuous automation with short history inside the docker container:
    - At most two images per model call (previous + current)
    - Last two actions as text for reasoning
    - Early stop when model emits finished
    """
    print(f"=== Step {step_idx} Started ===")
    print(f"Instruction: {instruction}")
    print(f"Max iterations: {max_iterations}")

    state = AutomationState(instruction=instruction, language="English")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        try:
            # 1) Capture current screenshot inside the docker container
            print("Capturing screenshot...")
            screenshot_bytes = await computer.interface.screenshot()
            cur_b64 = base64.b64encode(screenshot_bytes).decode("ascii")
            print("Screenshot captured and converted to base64")

            # 2) Build messages (<=2 images) and call model
            messages = build_messages_with_state(state, cur_b64)
            print("Calling UI grounding model with history...")
            raw_response = call_ui_grounding_model_with_messages(messages)

            # 3) Parse the response into actions
            structured_actions = parse_action_to_structure_output(
                raw_response,
                factor=FACTOR,
                origin_resized_height=image_height,
                origin_resized_width=image_width
            )

            # 4) Early stop if finished
            if any(a.get("action_type") == "finished" for a in structured_actions):
                print("Task completed (model emitted finished).")
                break

            # 5) Generate PyAutoGUI code
            pyautogui_code = parsing_response_to_pyautogui_code(
                structured_actions,
                image_height=screen_height,
                image_width=screen_width
            )
            print("--------------------------------")
            print("Generated PyAutoGUI code")
            print(pyautogui_code)
            print("--------------------------------")

            # Save every pyautogui code to a file for debugging
            os.makedirs("./data/automation_code", exist_ok=True)
            with open(f"./data/automation_code/automation_step_{step_idx}_{iteration + 1}.py", "w") as f:
                f.write(pyautogui_code)

            # 6) Execute code unless parser signaled DONE
            if pyautogui_code == "DONE":
                print("Task completed (parser signals DONE).")
                break

            print("Executing PyAutoGUI code...")
            await computer.interface.write_text("/tmp/my_script.py", pyautogui_code)
            result = await computer.interface.run_command(
                    "bash -lc 'timeout 15s python3 /tmp/my_script.py; rc=$?; pkill -f xclip || true; pkill -f xsel || true; echo __RC__$rc'"
                )
            print(f"Script executed with return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

            # 7) Visualize the actions using current image
            output_path = f"./data/screenshots/automation_step_{step_idx}_{iteration + 1}.png"
            visualize_actions_on_image(
                image=cur_b64,
                structured_actions=structured_actions,
                output_path=output_path,
                title=f"Automation Step {iteration + 1}: {instruction[:30]}..."
            )
            print(f"Actions visualized and saved to: {output_path}")

            # 8) Save step memory (previous image + last action summary)
            thought = structured_actions[0].get("thought", "") if structured_actions else ""
            first = structured_actions[0] if structured_actions else {"action_type": "", "action_inputs": {}}
            action_str = f"{first['action_type']}(" + ", ".join(
                f"{k}='{v}'" for k, v in first.get("action_inputs", {}).items()
            ) + ")"
            state.add_step(before_image_b64=cur_b64, thought=thought, action_str=action_str)

            # Short settle time
            time.sleep(1.0)

        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {str(e)}")
            break

    print("\n=== Step Ended ===")

async def main():
    # run_images_testing()

    # test_instructions_v1 = [
    #     "I need to click on the 'Member ID' input field.",\
    #     "Member ID input field is selected, I can type in 'E01257444' into the Member ID filed directly without clicking.",\
    #     "I need to click on the Month input field.",\
    #     "Month input field is selected, I can type in '11' into the Month input field directly without clicking.",\
    #     "I need to click on the Day input field.",\
    #     "Day input field is selected, I can type in '01' into the Day input field directly without clicking.",\
    #     "I need to click on the Year input field.",\
    #     "Year input field is selected, I can type in '1992' into the Year input field directly without clicking.",\
    #     "I need to click on the 'Search' button.",\
    #     "I need to click on 'Benefit Details'.",\
    #     "I need to click on the dropdown date range to open the file save dialog.",\
    #     "I need to click on the 'save' button.",\
    #     ]
    test_instructions_v2 = [
                "I need to click on the 'Member ID' input field and type in 'E01257444' into it.",\
                "I need to click on the 'Month/MM' input field and type in '11' into it.",\
                "I need to click on the 'Day/DD' input field and type in '01' into it.",\
                "I need to click on the 'Year/YYYY' input field and type in '1992' into it.",\
                "I need to click on the 'Search' button and click on 'Benefit Details'.",\
                "I need to click on the dropdown date range to open the file save dialog.",\
                "I need to click on the 'save' button to save the file.",\
            ]
    computer, image_width, image_height, screen_width, screen_height = await run_docker_container(image_name="cua-browser-ubuntu:latest")
    for i in range(len(test_instructions_v2)):
        await demo_docker_cua_step_automation(test_instructions_v2[i], computer, image_width, image_height, screen_width, screen_height, step_idx=i+1, max_iterations=3)
    
    await computer.stop()

    # # Test code integration
    # script = call_code_integration_model_from_dir("./data/automation_code")
    # print(script)
    # script = script.replace("```python", "").replace("```", "")

    # # Test integrated code execution
    # success, result = execute_pyautogui_code(script)
    # if success:
    #     print("Integrated code executed successfully")
    # else:
    #     print(f"Failed to execute integrated code: {result}")

    # # After execution: take current screenshot, load expected end image, and check result
    # try:
    #     current_b64 = get_screenshot_base64(width=RESIZED_MODEL_IMG_WIDTH, height=RESIZED_MODEL_IMG_HEIGHT)
    #     expected_path = './data/test_images/test_img_13.png'
    #     expected_image = Image.open(expected_path)
    #     _buf = BytesIO()
    #     expected_image.save(_buf, format="PNG")
    #     expected_b64 = base64.b64encode(_buf.getvalue()).decode()

    #     task_description = "Log in to the insurance portal and download the benefits details file."
    #     check = call_result_checking_model(
    #         task_description=task_description,
    #         expected_view_base64=expected_b64,
    #         current_view_base64=current_b64,
    #     )
    #     print("=== Result Checking ===")
    #     print(f"Thoughts: {check.get('thoughts', '')}")
    #     print(f"Finished: {check.get('result', False)}")
    # except Exception as e:
    #     print(f"Result checking failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())