import matplotlib.pyplot as plt
import os
import subprocess
import sys
import tempfile
from typing import Optional, Tuple, Union
import time


def visualize_actions_on_image(image: Union['PIL.Image.Image', str], structured_actions: list, output_path: str, title: Optional[str] = None, dpi: int = 350) -> None:
    """
    Visualize structured actions on an image and save it.

    Args:
        image: PIL Image object or base64 encoded image string to visualize actions on
        structured_actions: List of action dictionaries with 'action_type' and 'action_inputs'
        output_path: Path where to save the visualized image
        title: Optional title for the plot (will be truncated if too long)
        dpi: DPI for saving the image (default: 350)

    Returns:
        None (saves image to output_path)
    """
    # Handle both PIL Image objects and base64 strings
    if isinstance(image, str):
        # If it's a base64 string, convert to PIL Image
        try:
            from PIL import Image
            import base64
            from io import BytesIO

            # Decode base64 string to PIL Image
            image_data = base64.b64decode(image)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image string: {e}")
    elif not hasattr(image, 'size'):
        raise ValueError("Image must be a PIL Image object or base64 encoded string")

    # Ensure image is in RGB mode for proper matplotlib display
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_width, image_height = image.size
    
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
                        print(f"    Parsed {param_name}: coord_values={coord_values}, pixel_coords=({x}, {y})")
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
                        
                except (ValueError, AttributeError, TypeError) as e:
                    print(f"Error parsing coordinates for {param_name}: {param_value}, Error: {e}")
                    continue
        
        # After processing all parameters, draw drag line if applicable
        if action_type == 'drag' and len(coordinates) >= 2:
            if len(coordinates) == 2:  # We have both start and end
                start_x, start_y = coordinates[0]
                end_x, end_y = coordinates[1]
                plt.plot([start_x, end_x], [start_y, end_y], 'k--', alpha=0.5, linewidth=2)
                plt.annotate('drag path', ((start_x + end_x)/2, (start_y + end_y)/2), 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                           fontsize=7)
                print(f"  Drawn drag line from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        
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
    
    # Set title
    if title:
        # Truncate title if too long
        display_title = title[:50] + "..." if len(title) > 50 else title
        plt.title(f'Action Visualization - {display_title}')
    else:
        plt.title('Action Visualization')
    
    plt.axis('off')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image (without bbox_inches to preserve full image)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Visualization saved to: {output_path}")


def execute_pyautogui_code(code: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Execute generated pyautogui code safely.

    Args:
        code: The pyautogui code string to execute
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        Tuple[bool, str]: (success, error_message)
    """
    try:
        # Check if the code contains pyautogui imports
        if 'import pyautogui' not in code:
            return False, "Code must contain pyautogui import for safety"

        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Execute the code with timeout
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Clean up the temporary file
            os.unlink(temp_file_path)

            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            # Clean up the temporary file even if timeout occurs
            os.unlink(temp_file_path)
            return False, f"Execution timed out after {timeout} seconds"

    except Exception as e:
        return False, f"Failed to execute code: {str(e)}"


def get_screenshot_base64(region: Optional[Tuple[int, int, int, int]] = None, width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    Capture a screenshot and return it as a base64 encoded string.

    Args:
        region: Optional tuple (x, y, width, height) for region capture.
               If None, captures the entire screen.
        width: Optional target width to resize the screenshot to before encoding.
        height: Optional target height to resize the screenshot to before encoding.

    Returns:
        str: Base64 encoded string of the screenshot image
    """
    try:
        import pyautogui
        from PIL import Image
        import base64
        from io import BytesIO
    except ImportError:
        raise ImportError("pyautogui and pillow are required for screenshot capture. Install with: pip install pyautogui pillow")

    # Capture screenshot
    if region:
        screenshot = pyautogui.screenshot(region=region)
    else:
        screenshot = pyautogui.screenshot()

    # Resize if requested (no quality downscale, still PNG)
    if width is not None and height is not None:
        if screenshot.mode != 'RGB':
            screenshot = screenshot.convert('RGB')
        screenshot = screenshot.resize((int(width), int(height)))

    # Convert to base64 (PNG)
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


# Example usage for testing
if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)

    # Test the screenshot function
    print("Testing get_screenshot_base64...")
    try:
        base64_string = get_screenshot_base64()
        print(f"Screenshot captured successfully! Base64 length: {len(base64_string)}")
        print(f"First 100 characters: {base64_string[:100]}...")

        # Test visualization with base64 string
        print("\nTesting visualize_actions_on_image with base64 string...")

        # First, let's test the base64 decoding separately
        from PIL import Image
        import base64
        from io import BytesIO

        print("Decoding base64 to PIL Image...")
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(BytesIO(image_data))
        print(f"Original PIL Image mode: {pil_image.mode}")
        print(f"Original PIL Image size: {pil_image.size}")

        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            print(f"Converting from {pil_image.mode} to RGB...")
            pil_image = pil_image.convert('RGB')
            print(f"Converted PIL Image mode: {pil_image.mode}")

        # Use proper normalized coordinates (0-1 range) like real structured actions
        test_actions = [
            {
                "action_type": "click",
                "action_inputs": {
                    "start_box": [0.3, 0.1, 0.3, 0.1]  # Normalized coordinates (x1, y1, x2, y2)
                }
            },
            {
                "action_type": "type",
                "action_inputs": {"content": "Hello World"}
            }
        ]

        visualize_actions_on_image(
            image=base64_string,  # Pass base64 string directly!
            structured_actions=test_actions,
            output_path="./data/test_visualization.png",
            title="Test Visualization with Base64"
        )
        print("Visualization saved successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
