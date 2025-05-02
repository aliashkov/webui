import asyncio
import os
import shutil
import sys
import logging
import traceback
import time
import json
import random  # Added for randomization
from playwright.async_api import async_playwright
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from dotenv import load_dotenv  # Added for environment variable loading

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from emunium import EmuniumPlaywright
import psutil

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('browser_job.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def load_json_prompt(file_path: str = "prompts/test_prompt.json") -> tuple[str, str]:
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        logger.error(f"Prompt file '{file_path}' does not exist")
        return "", ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompt = data.get("prompt", "")
            add_infos = data.get("add_infos", "")
            if not prompt:
                logger.error(f"No 'prompt' field found in '{file_path}'")
                return "", ""
            logger.info(f"Successfully loaded prompt and add_infos from '{file_path}'")
            return prompt, add_infos
    except Exception as e:
        logger.error(f"Error loading prompt from '{file_path}': {e}")
        return "", ""

def terminate_chrome_process(cdp_port=9222):
    """Terminate all chrome.exe processes related to the CDP port."""
    terminated_pids = []
    try:
        for proc in psutil.process_iter(['name', 'cmdline', 'pid']):
            try:
                if proc.info['name'].lower() == 'chrome.exe':
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and f'--remote-debugging-port={cdp_port}' in cmdline:
                        proc.terminate()
                        terminated_pids.append(proc.info['pid'])
                        logger.info(f"Terminated Chrome process with PID {proc.info['pid']} (CDP-related)")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not terminate process {proc.info['pid']}: {e}")

        time.sleep(1)
        if not terminated_pids:
            logger.info("No Chrome processes found to terminate.")
    except Exception as e:
        logger.error(f"Error terminating Chrome processes: {e}")

async def close_browser_resources(browser: CustomBrowser, browser_context: CustomBrowserContext):
    try:
        if browser_context:
            await browser_context.close()
            logger.info("Browser context closed successfully.")
    except Exception as e:
        logger.error(f"Error closing browser context: {e}")
    finally:
        try:
            if browser:
                await browser.close()
                logger.info("Browser closed successfully.")
            else:
                logger.info("Browser already closed or not initialized.")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
        finally:
            terminate_chrome_process(cdp_port=9222)

async def run_browser_job(
    task: str,
    add_infos: str = "",
    cdp_url: str = "http://localhost:9222",
    window_w: int = 1400,
    window_h: int = 1025,
    max_steps: int = 2,
    max_actions_per_step: int = 3,
    use_vision: bool = False,
    enable_emunium: bool = True,
    keep_browser_open: bool = False,
    retry_delay: int = 25,
    max_attempts_per_task: int = 3,
    run_count: int = 1,
    enableEnter: bool = False
):
    """Run a browser job with retry mechanism, proxies, and stealth techniques."""
    attempt = 1
    global_browser = None
    global_browser_context = None
    global_agent = None
    
    PROXY_LIST = [
        "37.235.23.217:8080",
        "43.153.69.25:13001",
        "43.153.7.172:13001",
        "47.91.29.151:8080",
        "4.175.200.138:8080",
        "8.212.165.164:10801"
    ]

    while attempt <= max_attempts_per_task:
        logger.info(f"Attempt {attempt} of {max_attempts_per_task} for task")
        async with async_playwright() as p:
            try:
                # Step 1: Ensure Chrome is terminated before starting
                terminate_chrome_process(cdp_port=9222)
                await asyncio.sleep(2)

                # Step 2: Configure LLM with cycling API keys
                key_index = run_count % 6
                api_key_map = {
                    0: "GOOGLE_API_KEY",
                    1: "GOOGLE_API_KEY2",
                    2: "GOOGLE_API_KEY3",
                    3: "GOOGLE_API_KEY4",
                    4: "GOOGLE_API_KEY5",
                    5: "GOOGLE_API_KEY6",
                    6: "GOOGLE_API_KEY7"
                }
                api_key_name = api_key_map[key_index]
                api_key = os.getenv(api_key_name, "")
                if not api_key:
                    logger.error(f"{api_key_name} environment variable not set")
                    raise ValueError(f"{api_key_name} environment variable not set")
                llm = utils.get_llm_model(
                    provider="google",
                    model_name="gemini-2.0-flash",
                    temperature=0.6,
                    api_key=api_key
                )
                logger.info(f"Using {api_key_name} for run {run_count}")

                # Step 3: Initialize browser with proxy and stealth
                # Randomize window size
                window_w = random.randint(1500, 1700)
                window_h = random.randint(1225, 1425)
                extra_chromium_args = [
                    f"--window-size={window_w},{window_h}",
                ]
                
                proxy_server = random.choice(PROXY_LIST)

                # Load proxy settings from environment
                proxy = {
                    "server": f"http://{proxy_server}",
                    "username": "",  # Add if your proxies require auth
                    "password": ""   # Add if your proxies require auth
                }
                logger.info(f"Using proxy: {proxy_server}")
                
                cdp_url = os.getenv("CHROME_CDP", cdp_url)
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "":
                    chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data:
                    extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]

                browser_config = BrowserConfig(
                    headless=False,
                    disable_security=True,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                    proxy=proxy  # Add proxy to config
                )
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser initialized with CDP URL: {cdp_url}")

                # Step 4: Create browser context with stealth script
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp/traces") else None,
                        save_recording_path="./tmp/record_videos" if os.path.exists("./tmp/record_videos") else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h)
                    )
                )
                logger.info(f"Browser Context created: {global_browser_context}")

                # Step 5: Initialize controller and page
                controller = CustomController()
                page = await global_browser_context.get_current_page()
                emunium = EmuniumPlaywright(page)
                logger.info(f"Page initialized: {page}")

                # Simulate human-like behavior
                await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(random.uniform(1, 3))  # Random delay
                
                print("Enable Enter 2", enableEnter)

                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task,
                    add_infos=add_infos,  # Use add_infos correctly
                    use_vision=use_vision,
                    llm=llm,
                    browser=global_browser,
                    browser_context=global_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method="auto",
                    max_input_tokens=128000,
                    generate_gif=True
                )
                history = await global_agent.run(
                    max_steps=max_steps,
                    useOwnBrowser=True,
                    enable_emunium=True,
                    customHistory=True,
                    enableEnter=enableEnter
                )
                logger.info(f"Task completed successfully. Final Result: {history.final_result()}")

                # Step 7: Determine success status and save history/screenshots
                success = history.is_successful()  # Check if the task was successful
                logger.info(f"Task success status: {success}")

                # Define base directories
                base_history_dir = "./tmp/agent_history"
                base_screenshot_dir = "./tmp/screenshots"
                successful_history_dir = "./tmp/agent_history_successful"
                successful_screenshot_dir = "./tmp/screenshots_successful"

                # Create directories if they don't exist
                os.makedirs(base_history_dir, exist_ok=True)
                os.makedirs(base_screenshot_dir, exist_ok=True)
                if success:
                    os.makedirs(successful_history_dir, exist_ok=True)
                    os.makedirs(successful_screenshot_dir, exist_ok=True)

                # Define history file paths
                history_file = os.path.join(base_history_dir, f"{global_agent.state.agent_id}.json")
                successful_history_file = os.path.join(successful_history_dir, f"{global_agent.state.agent_id}.json") if success else None

                # Define screenshot directories (use run number from agent)
                run_screenshot_dir = os.path.join(base_screenshot_dir, f"run_{global_agent.run_number}")
                successful_run_screenshot_dir = os.path.join(successful_screenshot_dir, f"run_{global_agent.run_number}") if success else None
                os.makedirs(run_screenshot_dir, exist_ok=True)
                if success:
                    os.makedirs(successful_run_screenshot_dir, exist_ok=True)

                # Save history to agent_history
                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")

                # If successful, also save history to agent_history_successful
                if success:
                    global_agent.save_history(successful_history_file)
                    logger.info(f"Agent history also saved to {successful_history_file}")

                # Move screenshots to the base screenshots directory
                original_screenshot_dir = os.path.join("./tmp/screenshots", f"run_{global_agent.run_number}")
                if os.path.exists(original_screenshot_dir):
                    for filename in os.listdir(original_screenshot_dir):
                        src_path = os.path.join(original_screenshot_dir, filename)
                        dst_path = os.path.join(run_screenshot_dir, filename)
                        os.rename(src_path, dst_path)  # Move the screenshot
                        # If successful, also copy to screenshots_successful
                        if success:
                            successful_dst_path = os.path.join(successful_run_screenshot_dir, filename)
                            os.makedirs(os.path.dirname(successful_dst_path), exist_ok=True)
                            shutil.copy(dst_path, successful_dst_path)  # Use shutil.copy instead of os.copy
                            logger.info(f"Screenshot copied to {successful_dst_path}")
                    logger.info(f"Screenshots moved to {run_screenshot_dir}")
                    # Remove the original directory if empty
                    try:
                        os.rmdir(original_screenshot_dir)
                    except OSError:
                        pass  # Directory might not be empty or already removed

                return history.final_result()

            except Exception as e:
                error_msg = f"Error on attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                # Capture screenshot for debugging
                if global_browser_context:
                    page = await global_browser_context.get_current_page()
                    if page:
                        screenshot_path = f"./tmp/screenshot_run_{run_count}_attempt_{attempt}.png"
                        await page.screenshot(path=screenshot_path)
                        logger.info(f"Screenshot saved to {screenshot_path}")
                if attempt < max_attempts_per_task:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None
                    await asyncio.sleep(retry_delay)
                    attempt += 1
                else:
                    logger.error(f"Max attempts ({max_attempts_per_task}) reached for task.")
                    return None
            finally:
                if not keep_browser_open:
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None

    return None

async def main_loop():
    """Main loop to keep running tasks from a JSON prompt file with region-specific prompts."""
    task, add_infos = load_json_prompt(file_path="prompts/youtube_promotion_prompt3.json")
    if not task:
        logger.error("Failed to load task from JSON prompt file. Exiting.")
        return

    run_count = 1
    max_runs = 5000

    while max_runs is None or run_count < max_runs:
        run_count += 1
        logger.info(f"Starting run {run_count}")

        try:
            result = await run_browser_job(
                task=task,
                add_infos=add_infos,
                max_steps=200,
                max_actions_per_step=3,
                retry_delay=25,
                max_attempts_per_task=3,
                run_count=run_count,
                enableEnter=False
            )  # type: ignore
            if result:
                logger.info(f"Run {run_count} completed successfully with result: {result}")
            else:
                logger.warning(f"Run {run_count} failed after all attempts.")
            logger.info(f"Waiting 25 seconds before next run...")
            await asyncio.sleep(25)
        except Exception as e:
            logger.error(f"Unexpected error in run {run_count}: {str(e)}\n{traceback.format_exc()}")
            logger.info(f"Waiting 25 seconds before retrying...")
            await asyncio.sleep(25)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main loop: {str(e)}\n{traceback.format_exc()}")