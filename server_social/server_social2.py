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
                # Added check for proc.info being populated
                if proc.info and proc.info['name'] and proc.info['name'].lower() == 'chrome.exe':
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and f'--remote-debugging-port={cdp_port}' in cmdline:
                        proc.terminate()
                        terminated_pids.append(proc.info['pid'])
                        logger.info(f"Terminated Chrome process with PID {proc.info['pid']} (CDP-related)")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                # Check if proc.info exists before accessing pid
                pid_info = f"PID {proc.info['pid']}" if proc.info and 'pid' in proc.info else "unknown PID"
                logger.warning(f"Could not terminate process {pid_info}: {e}")
            except Exception as e: # Catch other potential errors during iteration
                 pid_info = f"PID {proc.info['pid']}" if proc.info and 'pid' in proc.info else "unknown PID"
                 logger.error(f"Error processing process {pid_info}: {e}")


        time.sleep(1) # Give processes time to terminate
        # Verify termination (optional but recommended)
        for pid in terminated_pids:
             if psutil.pid_exists(pid):
                 try:
                     proc = psutil.Process(pid)
                     proc.kill() # Force kill if terminate didn't work
                     logger.warning(f"Force killed lingering Chrome process with PID {pid}")
                 except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                     logger.info(f"Process {pid} terminated successfully or already gone: {e}")
                 except Exception as e:
                     logger.error(f"Error force killing process {pid}: {e}")

        if not terminated_pids:
            logger.info("No relevant Chrome processes found to terminate.")
    except Exception as e:
        logger.error(f"Error terminating Chrome processes: {e}")


async def close_browser_resources(browser: CustomBrowser | None, browser_context: CustomBrowserContext | None):
    # Ensure browser and context types are correct or handle potential None
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
            # Ensure Chrome processes are terminated even if browser close fails
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
    global_agent = None # Initialize agent to None

    # Define Proxy List (Consider moving to config or env vars)
    PROXY_LIST = os.getenv("PROXY_LIST", "37.235.23.217:8080,43.153.69.25:13001,43.153.7.172:13001,47.91.29.151:8080,4.175.200.138:8080,8.212.165.164:10801").split(',')
    if not PROXY_LIST or PROXY_LIST == ['']:
        logger.warning("PROXY_LIST environment variable not set or empty. Running without proxies.")
        PROXY_LIST = None


    while attempt <= max_attempts_per_task:
        logger.info(f"Attempt {attempt} of {max_attempts_per_task} for task (Run {run_count})")
        async with async_playwright() as p: # Playwright context manager ensures cleanup
            try:
                # Step 1: Ensure Chrome is terminated before starting
                logger.info("Terminating existing Chrome processes...")
                terminate_chrome_process(cdp_port=9222)
                await asyncio.sleep(2) # Short delay to ensure termination completes

                # Step 2: Configure LLM with cycling API keys
                # Ensure keys are loaded
                google_api_keys = [os.getenv(f"GOOGLE_API_KEY{i}") for i in range(1, 8)] # Assuming keys are named KEY, KEY2, ... KEY7
                google_api_keys = [key for key in google_api_keys if key] # Filter out None/empty keys

                if not google_api_keys:
                    logger.error("No GOOGLE_API_KEY environment variables found (expected GOOGLE_API_KEY, GOOGLE_API_KEY2, etc.).")
                    raise ValueError("No Google API keys configured.")

                key_index = (run_count - 1) % len(google_api_keys) # Use run_count (1-based) and modulo length
                api_key = google_api_keys[key_index]
                api_key_name = f"GOOGLE_API_KEY{key_index + 1 if key_index > 0 else ''}" # Construct name for logging

                logger.info(f"Using {api_key_name} for run {run_count}")

                llm = utils.get_llm_model(
                    provider="google",
                    model_name="gemini-1.5-flash", # Consider making model configurable
                    temperature=0.6,
                    api_key=api_key
                )


                # Step 3: Initialize browser with proxy and stealth
                # Randomize window size
                window_w = random.randint(1500, 1700)
                window_h = random.randint(1225, 1425)
                extra_chromium_args = [
                    f"--window-size={window_w},{window_h}"
                ]

                proxy_config = None
                if PROXY_LIST:
                    proxy_server = random.choice(PROXY_LIST)
                    proxy_config = {
                        "server": f"http://{proxy_server}", # Assuming HTTP proxies
                        # Add username/password if required by your proxies
                        # "username": os.getenv("PROXY_USERNAME"),
                        # "password": os.getenv("PROXY_PASSWORD")
                    }
                    logger.info(f"Using proxy: {proxy_server}")
                else:
                     logger.info("Running without proxy.")

                cdp_url = os.getenv("CHROME_CDP", cdp_url)
                chrome_path = os.getenv("CHROME_PATH") # Returns None if not set
                chrome_user_data = os.getenv("CHROME_USER_DATA")

                if chrome_user_data:
                    # Ensure user data dir exists or create it if desired
                    # os.makedirs(chrome_user_data, exist_ok=True)
                    extra_chromium_args.append(f"--user-data-dir={chrome_user_data}")
                    logger.info(f"Using Chrome user data dir: {chrome_user_data}")


                browser_config = BrowserConfig(
                    headless=False, # Consider making configurable
                    disable_security=True, # Use with caution
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path if chrome_path else None, # Pass None if empty string
                    extra_chromium_args=extra_chromium_args,
                    proxy=proxy_config
                )
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser connecting via CDP URL: {cdp_url}...")
                # Note: Connection happens implicitly when context is created or first action taken

                # Step 4: Create browser context with stealth script
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp/traces") else None,
                        save_recording_path="./tmp/record_videos" if os.path.exists("./tmp/record_videos") else None,
                        no_viewport=False, # Important for non-headless
                        browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                        # Add user agent override if needed
                        # user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
                    )
                )
                logger.info(f"Browser Context created: {global_browser_context}")

                # Apply stealth techniques like emunium
                page = await global_browser_context.get_current_page() # Get the initial page
                if not page:
                    logger.error("Failed to get initial page from browser context.")
                    raise ConnectionError("Could not get a page from the browser context.")


                logger.info(f"Page initialized: {page.url}") # Log initial URL


                # Step 5: Initialize controller
                controller = CustomController() # Assuming default initialization is fine

                # Simulate human-like behavior before agent starts
                try:
                    await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                    await asyncio.sleep(random.uniform(0.5, 1.5)) # Short pause after move
                    await page.evaluate(f"window.scrollBy(0, {random.randint(100, 300)})") # Small random scroll
                    await asyncio.sleep(random.uniform(1, 3))  # Random delay
                except Exception as interaction_err:
                    logger.warning(f"Minor error during initial human-like interaction simulation: {interaction_err}")


                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task,
                    add_infos=add_infos,
                    use_vision=use_vision,
                    llm=llm,
                    browser=global_browser,
                    browser_context=global_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method="auto",
                    max_input_tokens=128000, # Check model limits
                    generate_gif=True # Consider making configurable
                )
                logger.info(f"Agent created. Starting task run (EnableEnter={enableEnter})...")
                history = await global_agent.run(
                    max_steps=max_steps,
                    useOwnBrowser=True, # Assuming this means use the browser we set up
                    enable_emunium=False, # Already applied emunium above
                    customHistory=True, # Keep custom history handling
                    enableEnter=enableEnter
                )
                logger.info(f"Task execution finished. Final Result: {history.final_result()}")

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

                # Define history file paths using agent ID
                agent_id = global_agent.state.agent_id
                history_file = os.path.join(base_history_dir, f"{agent_id}.json")
                successful_history_file = os.path.join(successful_history_dir, f"{agent_id}.json") if success else None

                # Define screenshot directories using agent ID (more robust than run number)
                run_screenshot_dir = os.path.join(base_screenshot_dir, agent_id)
                successful_run_screenshot_dir = os.path.join(successful_screenshot_dir, agent_id) if success else None
                # Note: Screenshots are usually saved directly by the agent/browser actions into a dir related to agent_id

                # Save history
                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")
                if success and successful_history_file:
                    global_agent.save_history(successful_history_file)
                    logger.info(f"Successful agent history also saved to {successful_history_file}")


                # Move/Copy screenshots (Assuming they are initially saved in a standard location relative to the agent run)
                # The CustomAgent/Browser likely handles saving screenshots to a path like './tmp/screenshots/{agent_id}/...'
                # If they are saved elsewhere, adjust the 'original_screenshot_dir' logic.
                original_screenshot_dir = os.path.join("./tmp/screenshots", agent_id) # Default expected path

                if os.path.exists(original_screenshot_dir):
                     # If success, copy to successful dir first
                    if success and successful_run_screenshot_dir:
                        try:
                            shutil.copytree(original_screenshot_dir, successful_run_screenshot_dir, dirs_exist_ok=True)
                            logger.info(f"Screenshots copied to successful directory: {successful_run_screenshot_dir}")
                        except Exception as copy_err:
                            logger.error(f"Error copying screenshots to success directory: {copy_err}")

                    # Decide if you want to keep the base screenshots or remove them after copy
                    # To keep them (useful for debugging all runs):
                    logger.info(f"Screenshots available in base directory: {original_screenshot_dir}")
                    # To remove them if successful (save space):
                    # if success:
                    #     try:
                    #         shutil.rmtree(original_screenshot_dir)
                    #         logger.info(f"Removed original screenshots from {original_screenshot_dir} after successful copy.")
                    #     except Exception as rm_err:
                    #         logger.error(f"Error removing original screenshots: {rm_err}")

                else:
                     logger.warning(f"Screenshot directory not found at expected location: {original_screenshot_dir}")


                # If successful, break the retry loop
                return history.final_result()

            except Exception as e:
                error_msg = f"Error on attempt {attempt} (Run {run_count}): {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)

                # Capture screenshot for debugging if possible
                if global_browser_context:
                     try:
                        page = await global_browser_context.get_current_page()
                        if page and not page.is_closed():
                            screenshot_path = f"./tmp/error_screenshot_run_{run_count}_attempt_{attempt}.png"
                            os.makedirs("./tmp", exist_ok=True) # Ensure tmp dir exists
                            await page.screenshot(path=screenshot_path, full_page=True) # Capture full page on error
                            logger.info(f"Error screenshot saved to {screenshot_path}")
                        else:
                             logger.warning("Could not capture error screenshot: Page is closed or unavailable.")
                     except Exception as ss_err:
                         logger.error(f"Failed to take error screenshot: {ss_err}")


                attempt += 1 # Increment attempt counter
                if attempt <= max_attempts_per_task:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    # Close resources before retry
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None # Reset agent
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Max attempts ({max_attempts_per_task}) reached for task on run {run_count}.")
                    # Ensure resources are closed on final failure before returning None
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None
                    return None # Indicate failure after all attempts
            finally:
                # This block executes whether the try block succeeded or failed (after return or exception)
                # Close resources if 'keep_browser_open' is False and they haven't been closed already
                # (e.g., successful run or final failed attempt might have already closed them)
                if not keep_browser_open and (global_browser or global_browser_context):
                    logger.info(f"Closing browser resources for run {run_count} as keep_browser_open is False.")
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None


    # This should theoretically not be reached if the loop logic is correct,
    # but acts as a fallback return if the loop exits unexpectedly.
    logger.warning(f"Exited run_browser_job loop unexpectedly for run {run_count}.")
    await close_browser_resources(global_browser, global_browser_context) # Final cleanup attempt
    return None


async def main_loop():
    """Main loop to keep running tasks from a JSON prompt file with periodic long delays."""
    prompt_file = "prompts/youtube_promotion_prompt3.json" # Make easily configurable
    task, add_infos = load_json_prompt(file_path=prompt_file)
    if not task:
        logger.error(f"Failed to load task from '{prompt_file}'. Exiting.")
        return

    run_count = 0 # Start at 0, will be incremented to 1 before the first run
    max_runs = 5000 # Make configurable if needed

    # --- Delay Configuration ---
    SHORT_DELAY_SECONDS = 25
    LONG_DELAY_MINUTES = 10
    LONG_DELAY_SECONDS = LONG_DELAY_MINUTES * 60
    # Apply long delay every N runs (e.g., 20, 25)
    LONG_DELAY_INTERVAL = random.randint(20, 25) # Randomize interval slightly per script start
    logger.info(f"Configuration: Short delay={SHORT_DELAY_SECONDS}s. Long delay={LONG_DELAY_MINUTES}min every {LONG_DELAY_INTERVAL} runs.")
    # --- End Delay Configuration ---


    while max_runs is None or run_count < max_runs:
        run_count += 1 # Increment run counter for the current run
        logger.info(f"--- Starting Run {run_count}/{max_runs if max_runs else 'infinity'} ---")

        try:
            result = await run_browser_job(
                task=task,
                add_infos=add_infos,
                max_steps=40, # Make configurable if needed
                max_actions_per_step=3, # Make configurable if needed
                retry_delay=25, # Make configurable if needed
                max_attempts_per_task=3, # Make configurable if needed
                run_count=run_count,
                enableEnter=False, # Make configurable if needed
                use_vision=False # Make configurable if needed
            )
            if result:
                logger.info(f"Run {run_count} completed successfully. Result snippet: {str(result)[:100]}...") # Log snippet
            else:
                logger.warning(f"Run {run_count} failed after all attempts.")

        except Exception as e:
            logger.error(f"Critical unexpected error in main_loop during run {run_count}: {str(e)}\n{traceback.format_exc()}")
            # Decide if you want to stop the loop on critical errors or just wait and continue
            logger.info("A critical error occurred, applying short delay before attempting next run.")
            await asyncio.sleep(SHORT_DELAY_SECONDS) # Apply short delay even after critical error before next run

        finally:
            # --- Delay Logic ---
            # Check if it's time for a long delay (and not the very last run if max_runs is set)
            if run_count % LONG_DELAY_INTERVAL == 0 and (max_runs is None or run_count < max_runs):
                logger.info(f"Run {run_count} finished. Applying long delay ({LONG_DELAY_MINUTES} minutes) as run count is multiple of {LONG_DELAY_INTERVAL}.")
                await asyncio.sleep(LONG_DELAY_SECONDS)
                # Optional: Recalculate LONG_DELAY_INTERVAL for the next cycle
                LONG_DELAY_INTERVAL = random.randint(20, 25)
                logger.info(f"Next long delay will be after {LONG_DELAY_INTERVAL} more runs.")
            # Apply short delay if it's not the last run
            elif max_runs is None or run_count < max_runs:
                logger.info(f"Run {run_count} finished. Waiting {SHORT_DELAY_SECONDS} seconds before next run...")
                await asyncio.sleep(SHORT_DELAY_SECONDS)
            else:
                logger.info(f"Run {run_count} was the last run ({max_runs}). No further delay needed.")
            # --- End Delay Logic ---


    logger.info("Main loop finished after reaching max runs or stopping.")


if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs("./tmp/agent_history", exist_ok=True)
    os.makedirs("./tmp/screenshots", exist_ok=True)
    os.makedirs("./tmp/agent_history_successful", exist_ok=True)
    os.makedirs("./tmp/screenshots_successful", exist_ok=True)
    os.makedirs("./tmp/traces", exist_ok=True)
    os.makedirs("./tmp/record_videos", exist_ok=True)

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Critical error preventing main loop execution: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info("Script exit.")
        # Attempt a final cleanup of any lingering Chrome processes
        terminate_chrome_process(cdp_port=9222)