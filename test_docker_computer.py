import asyncio
import io
import os
import time
import re
from agent import ComputerAgent
from computer import Computer
from dotenv import load_dotenv
from utils import get_size_from_base64

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or ""
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY") or ""
os.environ["TGI_BASE_URL"] = os.getenv("TGI_BASE_URL") or ""

IMAGE = "cua-browser-ubuntu:latest"

async def main():
    """
    Replace main to execute all automation snippets inside a single Docker container.
    """
    import glob
    import pathlib

    computer = Computer(
        os_type="linux",
        provider_type="docker",
        image=IMAGE,
        name="my-cua-container",
        noVNC_port=6901,
        port=8000
    )

    await computer.run()

    print("VNC Access: http://localhost:6901")
    print("API: http://localhost:8000")

    # # Open the target webpage inside the container
    # _ = await computer.interface.run_command(
    #     "bash -lc 'nohup xdg-open https://www.brmsprovidergateway.com/provideronline/search.aspx >/dev/null 2>&1 </dev/null &'"
    # )
    # print(f"Browser open return code: {_.returncode}")
    # if _.stderr:
    #     print(f"Browser open STDERR: {_.stderr}")
    # time.sleep(5)

    try:
        dir_path = pathlib.Path("./data/automation_code").resolve()
        pattern = str(dir_path / "automation_step_*.py")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"No automation snippets found under: {dir_path}")
            return

        print(f"Found {len(files)} snippet(s). Executing in container...\n")

        os.makedirs("./data/screenshots", exist_ok=True)

        for idx, file_path in enumerate(files, 1):
            print(f"=== [{idx}/{len(files)}] Executing: {file_path} ===")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                # Write and execute inside the container with timeout and cleanup
                await computer.interface.write_text("/tmp/my_script.py", code)
                result = await computer.interface.run_command(
                    "bash -lc 'timeout 15s python3 /tmp/my_script.py; rc=$?; pkill -f xclip || true; pkill -f xsel || true; echo __RC__$rc'"
                )

                print(f"Return code: {result.returncode}")
                snippet_rc = None
                if result.stdout:
                    print(f"STDOUT:\n{result.stdout}")
                    m = re.search(r"__RC__(\d+)", result.stdout)
                    if m:
                        snippet_rc = int(m.group(1))
                if snippet_rc is not None:
                    if snippet_rc == 0:
                        print("Snippet status: completed successfully")
                    elif snippet_rc == 124:
                        print("Snippet status: timed out (timeout(1) exit code 124)")
                    else:
                        print(f"Snippet status: exited with code {snippet_rc}")
                if result.stderr:
                    print(f"STDERR:\n{result.stderr}")

                # Save a screenshot after each snippet
                screenshot_bytes = await computer.interface.screenshot()
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                out_path = f"./data/screenshots/{base_name}_{ts}.png"
                with open(out_path, "wb") as out:
                    out.write(screenshot_bytes)
                print(f"Screenshot saved: {out_path}\n")

                time.sleep(0.5)
            except Exception as e:
                print(f"Error executing {file_path}: {e}\n")
                # Continue with next snippet
                continue
    finally:
        # await computer.stop()
        pass


if __name__ == "__main__":
    asyncio.run(main())