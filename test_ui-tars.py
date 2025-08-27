# pip install openai
import io
import os
import re
import json
import base64
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from action_parser import add_box_token, parse_action_to_structure_output, parsing_response_to_pyautogui_code, smart_resize, parse_action, convert_point_to_coordinates
from prompts.prompts import COMPUTER_USE_DOUBAO

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or ""
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY") or ""
os.environ["TGI_BASE_URL"] = os.getenv("TGI_BASE_URL") or ""

MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT = 3360, 2100
SCREEN_WIDTH, SCREEN_HEIGHT = 1680, 1050
FACTOR = 28


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

client = OpenAI(
    base_url=os.environ["TGI_BASE_URL"],
    api_key=os.environ["HF_TOKEN"]
)

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
    
    # Update test input with current instruction and image
    messages = [
        {
            "role": "user", 
            "content": COMPUTER_USE_DOUBAO.format(instruction=cur_test_instruction, language="English")
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_str}"
                    }
                }
            ]
        }
    ]

    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        top_p=None,
        temperature=0.0,
        max_tokens=400,
        stream=True,   
        seed=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None
    )

    raw_chunks = []
    raw_response = ""
    for message in chat_completion:
        raw_chunks.append(message)
        delta = message.choices[0].delta.content or ""
        raw_response += delta

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

    # Display the image
    plt.clf()
    plt.imshow(image)
    
    # Process each action and visualize it
    for action_idx, action in enumerate(structured_actions):
        action_type = action['action_type']
        action_inputs = action['action_inputs']
        
        # Check if action has coordinates
        has_coordinates = False
        coordinates = []
        
        # Check for coordinate-based parameters
        for param_name, param_value in action_inputs.items():
            if any(coord_param in param_name for coord_param in ['start_box', 'end_box', 'point']):
                try:
                    # Handle different coordinate formats
                    if isinstance(param_value, str):
                        # Remove brackets and parse
                        coord_str = param_value.strip('[]')
                        coord_values = [float(x.strip()) for x in coord_str.split(',')]
                    else:
                        coord_values = param_value
                    
                    if len(coord_values) >= 2:
                        # Convert normalized coordinates (0-1) to pixel coordinates
                        x = int(coord_values[0] * image_width)
                        y = int(coord_values[1] * image_height)
                        coordinates.append((x, y))
                        has_coordinates = True
                        
                        # Choose color based on parameter type
                        if 'start' in param_name:
                            color = 'red'
                            marker_size = 100
                        elif 'end' in param_name:
                            color = 'blue'
                            marker_size = 100
                        else:
                            color = 'green'
                            marker_size = 80
                        
                        # Draw circle for coordinate
                        plt.scatter([x], [y], c=color, s=marker_size, alpha=0.7, edgecolors='black', linewidth=2)
                        
                        # Add action type label near the coordinate
                        label_text = f'{action_type}'
                        if param_name != 'start_box':  # Add parameter name if not default
                            label_text += f'({param_name})'
                        
                        plt.annotate(label_text, (x, y), 
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                   fontsize=8, fontweight='bold')
                        
                        print(f"  Drawn coordinate: ({x}, {y}) for {action_type} ({param_name})")
                        
                        # If this is a drag action and we have both start and end points, draw a line
                        if action_type == 'drag' and len(coordinates) >= 2:
                            if len(coordinates) == 2:  # We have both start and end
                                start_x, start_y = coordinates[0]
                                end_x, end_y = coordinates[1]
                                plt.plot([start_x, end_x], [start_y, end_y], 'k--', alpha=0.5, linewidth=2)
                                plt.annotate('drag path', ((start_x + end_x)/2, (start_y + end_y)/2), 
                                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                                           fontsize=7)
                        
                except (ValueError, AttributeError, TypeError) as e:
                    print(f"Error parsing coordinates for {param_name}: {param_value}, Error: {e}")
                    continue
        
        # If no coordinates found, add text annotation for the action
        if not has_coordinates:
            # Add text annotation for non-coordinate actions
            action_text = f"{action_type}"
            
            # Add relevant parameters to the text
            param_texts = []
            for param_name, param_value in action_inputs.items():
                if param_name == 'content':
                    content = str(param_value)
                    # Truncate content if too long
                    if len(content) > 30:
                        content = content[:27] + "..."
                    param_texts.append(f"content: '{content}'")
                elif param_name == 'key':
                    param_texts.append(f"key: '{param_value}'")
                elif param_name == 'direction':
                    param_texts.append(f"direction: '{param_value}'")
                else:
                    param_texts.append(f"{param_name}: '{param_value}'")
            
            if param_texts:
                action_text += f"({', '.join(param_texts)})"
            
            # Position text in the top-left corner with offset for multiple actions
            text_x = 50
            text_y = 50 + (action_idx * 40)  # Increased spacing
            
            plt.annotate(action_text, (text_x, text_y), 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                        fontsize=9, fontweight='bold')
            
            print(f"  Added text annotation: {action_text}")
    
    plt.title(f'Action Visualization - {cur_test_instruction[:50]}...')
    plt.axis('off')
    plt.savefig(f'./data/test_images/coordinate_process_image_{i+1}.png', dpi=350, bbox_inches='tight')
    plt.close()

    # pyautogui_code = parsing_response_to_pyautogui_code(
    #     structured_actions,
    #     image_height=SCREEN_HEIGHT,
    #     image_width=SCREEN_WIDTH
    # )

    # print("=== Generated PyAutoGUI code ===")
    # print(pyautogui_code)

    # break