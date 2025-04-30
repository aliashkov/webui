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
    window_w: int = 1280,
    window_h: int = 1025,
    max_steps: int = 2,
    max_actions_per_step: int = 3,
    use_vision: bool = False,
    enable_emunium: bool = True,
    keep_browser_open: bool = False,
    retry_delay: int = 25,
    max_attempts_per_task: int = 3,
    run_count: int = 1
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
        logger.info(f"Starting Task Run {run_count}, Attempt {attempt}/{max_attempts_per_task}")
        try:
            async with async_playwright() as p:
                # Step 1: Ensure Chrome is terminated before starting
                logger.info("Ensuring previous Chrome instances are terminated...")
                terminate_chrome_process(cdp_port=9222)
                await asyncio.sleep(random.uniform(1.5, 3.0)) # Slightly longer, randomized wait

                # Step 2: Configure LLM with cycling API keys
                # Cycle through 7 keys (index 0 to 6)
                key_index = (run_count -1) % 7 # Use run_count - 1 for 0-based index
                api_key_name = f"GOOGLE_API_KEY{key_index + 1}" if key_index > 0 else "GOOGLE_API_KEY" # KEY, KEY2, ... KEY7
                api_key = os.getenv(api_key_name)
                if not api_key:
                    logger.error(f"{api_key_name} environment variable not set")
                    # Fallback or raise error - decide your strategy
                    # Option 1: Try the default key
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                         raise ValueError(f"Primary GOOGLE_API_KEY and specific {api_key_name} environment variables not set")
                    else:
                         logger.warning(f"{api_key_name} not set, falling back to GOOGLE_API_KEY")
                         api_key_name = "GOOGLE_API_KEY" # Update name for logging
                    # Option 2: Raise error immediately (current behavior)
                    # raise ValueError(f"{api_key_name} environment variable not set")

                llm = utils.get_llm_model(
                    provider="google",
                    model_name="gemini-1.5-flash", # Consider flash or pro based on needs
                    temperature=random.uniform(0.5, 0.8), # Randomize temperature slightly
                    api_key=api_key
                )
                logger.info(f"Using LLM with API Key: {api_key_name} for Run {run_count}")

                # Step 3: Initialize browser with proxy and stealth
                # Randomize window size within a range
                current_window_w = random.randint(1250, 1450)
                current_window_h = random.randint(1000, 1200)
                logger.info(f"Using window size: {current_window_w}x{current_window_h}")

                extra_chromium_args = [
                    f"--window-size={current_window_w},{current_window_h}"
                ]

                # Select a random proxy for this attempt
                proxy_server = random.choice(PROXY_LIST)
                proxy_config = {
                    "server": f"http://{proxy_server}",
                    "username": os.getenv("PROXY_USERNAME", ""), # Load credentials from env if they exist
                    "password": os.getenv("PROXY_PASSWORD", "")
                }
                logger.info(f"Using proxy: {proxy_server}")

                # Get browser config from environment variables
                cdp_url_env = os.getenv("CHROME_CDP", cdp_url) # Use default if not set
                chrome_path = os.getenv("CHROME_PATH") or None # Handles empty string or None
                chrome_user_data = os.getenv("CHROME_USER_DATA") or None # Handles empty string or None

                if chrome_user_data:
                    extra_chromium_args.append(f"--user-data-dir={chrome_user_data}")
                    logger.info(f"Using Chrome user data dir: {chrome_user_data}")
                else:
                     logger.info("Not using a persistent Chrome user data directory.")


                browser_config = BrowserConfig(
                    headless=False,
                    disable_security=True,
                    cdp_url=cdp_url_env,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                    proxy=proxy_config
                )
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser attempting connection via CDP URL: {cdp_url_env}")

                # It might take a moment for the browser launched via CDP to be ready
                await asyncio.sleep(random.uniform(2, 4))

                # Step 4: Create browser context
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp") else None,
                        save_recording_path="./tmp/record_videos" if os.path.exists("./tmp") else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(width=current_window_w, height=current_window_h),
                        # Add other context options like locale, timezone etc. for stealth
                        locale="en-US"                    )
                )
                logger.info(f"Browser Context created: {global_browser_context}")

                # Step 5: Initialize controller and page
                controller = CustomController()
                page = await global_browser_context.get_current_page()
                if not page:
                     logger.error("Failed to get current page from browser context.")
                     raise ConnectionError("Failed to get page from browser context")


                if enable_emunium:
                     try:
                         emunium = EmuniumPlaywright(page)
                         # await emunium.smell_like_a_human() # Example Emunium usage
                         logger.info("Emunium initialized (if enabled).")
                     except Exception as emu_err:
                         logger.warning(f"Failed to initialize or use Emunium: {emu_err}")


                logger.info(f"Page obtained: {page.url}") # Log initial URL

                # Optional: Add initial human-like interaction
                await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                await asyncio.sleep(random.uniform(0.5, 1.5))
                await page.evaluate(f"window.scrollBy(0, {random.randint(50, 200)})")
                await asyncio.sleep(random.uniform(1, 3))

                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task, # The dynamically generated task prompt
                    add_infos=add_infos, # The dynamically generated add_infos
                    use_vision=use_vision,
                    llm=llm,
                    browser=global_browser,
                    browser_context=global_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method="auto",
                    max_input_tokens=128000, # Check model context limits
                    generate_gif=True # Keep generating gifs if needed
                )

                logger.info(f"Running agent for Task Run {run_count}...")
                history = await global_agent.run(
                    max_steps=max_steps
                )
                final_result = history.final_result()
                logger.info(f"Task Run {run_count} completed. Final Result: {final_result}")

                # Step 7: Determine success status and save history/screenshots
                success = history.is_successful()
                logger.info(f"Task Run {run_count} success status: {success}")

                # Define base directories more robustly
                base_dir = os.path.abspath("./tmp")
                base_history_dir = os.path.join(base_dir, "agent_history")
                base_screenshot_dir = os.path.join(base_dir, "screenshots")
                successful_history_dir = os.path.join(base_dir, "agent_history_successful")
                successful_screenshot_dir = os.path.join(base_dir, "screenshots_successful")

                # Create directories if they don't exist
                os.makedirs(base_history_dir, exist_ok=True)
                os.makedirs(base_screenshot_dir, exist_ok=True)
                if success:
                    os.makedirs(successful_history_dir, exist_ok=True)
                    os.makedirs(successful_screenshot_dir, exist_ok=True)

                # Use agent_id for unique filenames
                agent_id = global_agent.state.agent_id
                history_file = os.path.join(base_history_dir, f"{agent_id}.json")
                successful_history_file = os.path.join(successful_history_dir, f"{agent_id}.json") if success else None

                # Define screenshot directories using agent_id for clarity
                run_screenshot_dir = os.path.join(base_screenshot_dir, agent_id) # Use agent_id subfolder
                successful_run_screenshot_dir = os.path.join(successful_screenshot_dir, agent_id) if success else None
                # Screenshots are likely saved by the agent already in ./tmp/screenshots/<agent_id>
                # We just need to potentially copy them if successful

                # Save history
                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")
                if success and successful_history_file:
                    global_agent.save_history(successful_history_file)
                    logger.info(f"Successful agent history also saved to {successful_history_file}")

                # Handle screenshots (Copy successful ones)
                original_agent_screenshot_dir = os.path.join(base_screenshot_dir, agent_id) # Agent saves here by default
                if success and successful_run_screenshot_dir:
                    if os.path.exists(original_agent_screenshot_dir):
                        try:
                            # Copy the entire directory for successful runs
                            shutil.copytree(original_agent_screenshot_dir, successful_run_screenshot_dir, dirs_exist_ok=True)
                            logger.info(f"Screenshots for successful run {agent_id} copied to {successful_run_screenshot_dir}")
                        except Exception as copy_err:
                            logger.error(f"Error copying screenshots from {original_agent_screenshot_dir} to {successful_run_screenshot_dir}: {copy_err}")
                    else:
                        logger.warning(f"Original screenshot directory not found, cannot copy: {original_agent_screenshot_dir}")


                # If successful, break the retry loop
                return final_result

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
    task, add_infos = load_json_prompt(file_path="prompts/comments/gather_prompt4.json")
    if not task:
        logger.error("Failed to load task from JSON prompt file. Exiting.")
        return

    run_count = 1
    max_runs = 5000

    while max_runs is None or run_count < max_runs:
        run_count += 1
        logger.info(f"Starting run {run_count}")

        # Determine region based on run_count
        if 1 <= run_count <= 300:
            region = "Germany"
        elif 301 <= run_count <= 600:
            region = "England"
        elif 601 <= run_count <= 900:
            region = "Spain"
        elif 901 <= run_count <= 1200:
            region = "Italy"
        elif 1201 <= run_count <= 1500:
            region = "Netherlands"
        elif 1501 <= run_count <= 1800:
            region = "Belgia"
        elif 1801 <= run_count <= 2100:
            region = "France"
        else:
            region = "Unknown"

        # Append region to add_infos
        region_info = f"Country: {region}"
        modified_add_infos = f"{add_infos}\n{region_info}" if add_infos else region_info

        try:
            result = await run_browser_job(
                task=f"Click to {run_count} page" + region_info + task,
                add_infos=add_infos,
                max_steps=200,
                max_actions_per_step=3,
                retry_delay=25,
                max_attempts_per_task=3,
                run_count=run_count
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