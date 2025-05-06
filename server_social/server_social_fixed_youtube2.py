import asyncio
import os
import shutil
import sys
import logging
import traceback
import time
import json
import random
from playwright.async_api import async_playwright
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from dotenv import load_dotenv
import psutil

# Project root and path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from emunium import EmuniumPlaywright

# Load environment variables
load_dotenv()

# Configuration Constants
CONFIG = {
    "RETRY_DELAY": 25,  # Delay between retry attempts in seconds
    "ENABLE_EMUNIUM": True,  # Enable Emunium for browser automation
    "MAX_STEPS": 180,  # Increased: Maximum steps for the agent to complete the task (was 40)
    "MAX_ACTIONS_PER_STEP": 3,  # Maximum actions per step
    "MAX_ATTEMPTS_PER_TASK": 3,  # Maximum retry attempts per task
    "ENABLE_ENTER": False,  # Enable Enter key simulation
    "CDP_PORT": 9222,  # Chrome DevTools Protocol port
    "WINDOW_WIDTH": 1500,  # Default window width
    "WINDOW_HEIGHT": 1025,  # Default window height
    "KEEP_BROWSER_OPEN": False,  # Whether to keep browser open after task
    "USE_VISION": False,  # Enable vision-based processing
    "MAX_RUNS": 5000,  # Maximum number of runs in main loop
    "USE_OWN_BROWSER": True, # Enable custom browser
    "ENABLE_CLICK": True,
    "PROXY_LIST": [
        "37.235.23.217:8080",
        "43.153.69.25:13001",
        "43.153.7.172:13001",
        "47.91.29.151:8080",
        "4.175.200.138:8080",
        "8.212.165.164:10801"
    ]
}

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

def terminate_chrome_process(cdp_port: int = CONFIG["CDP_PORT"]):
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
            terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])

async def run_browser_job(
    task: str,
    add_infos: str = "",
    cdp_url: str = "http://localhost:9222",
    window_w: int = CONFIG["WINDOW_WIDTH"],
    window_h: int = CONFIG["WINDOW_HEIGHT"],
    run_count: int = 1,
):
    """Run a browser job with retry mechanism, proxies, and stealth techniques."""
    attempt = 1
    global_browser = None
    global_browser_context = None
    global_agent = None
    
    while attempt <= CONFIG["MAX_ATTEMPTS_PER_TASK"]:
        logger.info(f"Attempt {attempt} of {CONFIG['MAX_ATTEMPTS_PER_TASK']} for task")
        async with async_playwright() as p:
            try:
                # Step 1: Ensure Chrome is terminated before starting
                terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])
                await asyncio.sleep(2)

                # Step 2: Configure LLM with cycling API keys
                # api_key_map has 7 entries (indices 0-6)
                key_index = run_count % 7 # Corrected from % 6
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
                    model_name="gemini-2.0-flash", # Consider gemini-1.5-flash if available & compatible
                    temperature=0.6,
                    api_key=api_key
                )
                logger.info(f"Using {api_key_name} for run {run_count}")

                # Step 3: Initialize browser with proxy and stealth
                # Randomize window size
                window_w = random.randint(CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_WIDTH"] + 200)
                window_h = random.randint(CONFIG["WINDOW_HEIGHT"], CONFIG["WINDOW_HEIGHT"] + 200)
                extra_chromium_args = [
                    f"--window-size={window_w},{window_h}",
                ]
                
                proxy_server = random.choice(CONFIG["PROXY_LIST"])
                proxy = {
                    "server": f"http://{proxy_server}",
                    "username": "", 
                    "password": ""  
                }
                logger.info(f"Using proxy: {proxy_server}")
                
                cdp_url_env = os.getenv("CHROME_CDP", cdp_url) # Use a different variable name to avoid confusion with function param
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "": # Ensure empty string becomes None
                    chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data: # Ensure empty string becomes None or handle properly
                    if chrome_user_data == "":
                        chrome_user_data = None
                    else:
                         extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]


                browser_config = BrowserConfig(
                    headless=False,
                    disable_security=True,
                    cdp_url=cdp_url_env, # Use the potentially env-overridden cdp_url
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                    proxy=proxy
                )
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser initialized with CDP URL: {cdp_url_env}")

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
                await asyncio.sleep(random.uniform(1, 3))
                
                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task,
                    add_infos=add_infos,
                    use_vision=CONFIG["USE_VISION"],
                    llm=llm,
                    browser=global_browser,
                    browser_context=global_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=CONFIG["MAX_ACTIONS_PER_STEP"],
                    tool_calling_method="auto",
                    max_input_tokens=128000, # For Gemini 1.5 Flash, this can be much higher (e.g., 1M or 2M if needed)
                    generate_gif=True
                )
                history = await global_agent.run(
                    max_steps=CONFIG["MAX_STEPS"],
                    useOwnBrowser=CONFIG["USE_OWN_BROWSER"],
                    enable_emunium=CONFIG["ENABLE_EMUNIUM"],
                    customHistory=True,
                    enableEnter=CONFIG["ENABLE_ENTER"],
                    enableClick=CONFIG["ENABLE_CLICK"]
                )
                logger.info(f"Task completed successfully. Final Result: {history.final_result()}")

                # Step 7: Determine success status and save history/screenshots
                success = history.is_successful()
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

                # Define screenshot directories
                # Screenshots are already saved within agent.run() if generate_gif is true or by steps
                # The following logic renames/moves based on `global_agent.run_number` which might be an internal counter.
                # Ensure screenshots are actually being placed in "./tmp/screenshots/run_{run_number}" by the agent or its tools.
                # If screenshots are saved elsewhere by the agent's internal logic, this move needs to be adjusted.
                # Assuming screenshots are in a subfolder named after agent_id or similar:
                
                agent_screenshot_source_dir = os.path.join("./tmp/screenshots", global_agent.state.agent_id) # Example if agent saves to its ID
                # If the agent directly saves to "run_{run_number}" then original_screenshot_dir is correct
                original_screenshot_dir = os.path.join("./tmp/screenshots", f"run_{global_agent.run_number}")


                run_screenshot_target_dir = os.path.join(base_screenshot_dir, f"{global_agent.state.agent_id}_run_{run_count}") # More specific naming
                successful_run_screenshot_target_dir = os.path.join(successful_screenshot_dir, f"{global_agent.state.agent_id}_run_{run_count}") if success else None
                
                os.makedirs(run_screenshot_target_dir, exist_ok=True)
                if success and successful_run_screenshot_target_dir:
                    os.makedirs(successful_run_screenshot_target_dir, exist_ok=True)


                # Save history
                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")
                if success and successful_history_file:
                    global_agent.save_history(successful_history_file) # This might be redundant if history.is_successful() is used for the initial save decision
                    logger.info(f"Agent history also saved to {successful_history_file}")

                # Move/Copy screenshots (adjust source path if necessary)
                # This part assumes screenshots from the current run are in a specific `original_screenshot_dir`
                # If your agent's `generate_gif` or screenshot logic saves them directly to a path including `agent_id`,
                # you might need to adjust `original_screenshot_dir`.
                # For now, let's assume the agent saves screenshots to a directory identifiable by `global_agent.state.agent_id` or `global_agent.run_number`

                # Check if the agent's default screenshot directory exists
                # This path might need to be confirmed based on how CustomAgent saves screenshots
                source_screenshots_path = os.path.join("./tmp", "screenshots", global_agent.state.agent_id) 
                if not os.path.exists(source_screenshots_path): # Fallback if agent saves by run_number
                    source_screenshots_path = os.path.join("./tmp", "screenshots", f"run_{global_agent.run_number}")

                if os.path.exists(source_screenshots_path):
                    # Create target directories
                    os.makedirs(run_screenshot_target_dir, exist_ok=True)
                    if success and successful_run_screenshot_target_dir:
                         os.makedirs(successful_run_screenshot_target_dir, exist_ok=True)

                    for filename in os.listdir(source_screenshots_path):
                        src_path = os.path.join(source_screenshots_path, filename)
                        dst_path_main = os.path.join(run_screenshot_target_dir, filename)
                        
                        try:
                            # shutil.move(src_path, dst_path_main) # Use copy if you want to keep originals or for debugging
                            shutil.copy(src_path, dst_path_main)
                            logger.info(f"Screenshot copied to {dst_path_main}")
                            if success and successful_run_screenshot_target_dir:
                                dst_path_successful = os.path.join(successful_run_screenshot_target_dir, filename)
                                shutil.copy(src_path, dst_path_successful)
                                logger.info(f"Screenshot also copied to {dst_path_successful}")
                        except Exception as e_mv:
                            logger.error(f"Error moving/copying screenshot {src_path}: {e_mv}")
                    
                    # Optionally remove the source_screenshots_path if it's a temporary holding folder per run
                    # Be cautious with rmdir or rmtree if other runs might use it or if it's a parent folder
                    # If screenshots are directly generated into agent_id or run_number specific folders,
                    # this "move" might be more about "copying to successful" and "organizing" than moving from a generic spot.

                else:
                    logger.warning(f"Screenshot source directory {source_screenshots_path} not found for agent {global_agent.state.agent_id}.")


                return history.final_result()

            except Exception as e:
                error_msg = f"Error on attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                if global_browser_context:
                    try:
                        page = await global_browser_context.get_current_page()
                        if page and not page.is_closed(): # Check if page is not None and not closed
                            screenshot_path = f"./tmp/screenshot_run_{run_count}_attempt_{attempt}.png"
                            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True) # Ensure ./tmp exists
                            await page.screenshot(path=screenshot_path)
                            logger.info(f"Screenshot saved to {screenshot_path}")
                        elif page and page.is_closed():
                            logger.warning("Page was closed, cannot take error screenshot.")
                        else:
                            logger.warning("Page object not available, cannot take error screenshot.")
                    except Exception as se:
                        logger.error(f"Failed to take error screenshot: {se}")

                if attempt < CONFIG["MAX_ATTEMPTS_PER_TASK"]:
                    logger.info(f"Retrying in {CONFIG['RETRY_DELAY']} seconds...")
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None
                    await asyncio.sleep(CONFIG["RETRY_DELAY"])
                    attempt += 1
                else:
                    logger.error(f"Max attempts ({CONFIG['MAX_ATTEMPTS_PER_TASK']}) reached for task.")
                    return None # Ensure this path returns None
            finally:
                if not CONFIG["KEEP_BROWSER_OPEN"] or attempt > CONFIG["MAX_ATTEMPTS_PER_TASK"]: # Ensure closure on max attempts fail
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None # Nullify to prevent reuse
                    global_browser_context = None
                    global_agent = None
        # Ensure playwright instance `p` is exited cleanly if loop continues due to retry
        # The `async with async_playwright() as p:` handles this per attempt.
    logger.warning(f"Task failed after {CONFIG['MAX_ATTEMPTS_PER_TASK']} attempts.") # Log if all attempts failed
    return None


async def main_loop():
    """Main loop to keep running tasks from a JSON prompt file."""
    # IMPORTANT: Create a new JSON file (e.g., prompts/youtube_multi_promo_prompt.json)
    # with the content provided in step 2, and update the file_path here.
    task, add_infos = load_json_prompt(file_path="prompts/youtube_multi_promo_prompt.json") # MODIFIED_HERE
    if not task:
        logger.error("Failed to load task from JSON prompt file. Exiting.")
        return

    run_count = 0 # Start run_count from 0 if you want the first run to be 1 after increment

    while CONFIG["MAX_RUNS"] is None or run_count < CONFIG["MAX_RUNS"]:
        run_count += 1 # Increment at the start of the loop
        logger.info(f"Starting run {run_count} of {CONFIG['MAX_RUNS'] if CONFIG['MAX_RUNS'] else 'unlimited'}")

        try:
            result = await run_browser_job(
                task=task,
                add_infos=add_infos,
                run_count=run_count,
            )
            if result:
                logger.info(f"Run {run_count} completed. Result: {result}")
            else:
                logger.warning(f"Run {run_count} failed or returned no result after all attempts.")
            
            # Delay before next run, regardless of success or failure of the previous one
            logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before next run (if applicable)...")
            await asyncio.sleep(CONFIG["RETRY_DELAY"])

        except Exception as e:
            logger.error(f"Unexpected error in main_loop during run {run_count}: {str(e)}\n{traceback.format_exc()}")
            logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before attempting next run...")
            await asyncio.sleep(CONFIG["RETRY_DELAY"]) # Wait before retrying the loop

if __name__ == "__main__":
    # Create tmp directories if they don't exist
    os.makedirs("./tmp/agent_history", exist_ok=True)
    os.makedirs("./tmp/screenshots", exist_ok=True)
    os.makedirs("./tmp/agent_history_successful", exist_ok=True)
    os.makedirs("./tmp/screenshots_successful", exist_ok=True)
    os.makedirs("./tmp/traces", exist_ok=True)
    os.makedirs("./tmp/record_videos", exist_ok=True)
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main application: {str(e)}\n{traceback.format_exc()}")
    finally:
        # Final cleanup, though terminate_chrome_process is called within run_browser_job
        logger.info("Main loop finished or terminated. Performing final cleanup.")
        terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])