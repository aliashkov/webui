import asyncio
import os
import sys
import logging
import traceback
import time
import json
import random
import shutil # Added for copying successful screenshots
from playwright.async_api import async_playwright
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from dotenv import load_dotenv
import psutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from emunium import EmuniumPlaywright


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
            logger.info(f"Successfully loaded base prompt and add_infos from '{file_path}'")
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
                # Use lower() for case-insensitive comparison
                if proc.info['name'].lower() == 'chrome.exe':
                    cmdline = proc.info.get('cmdline', [])
                    # Check if cmdline is not None and is iterable
                    if cmdline and isinstance(cmdline, (list, tuple)) and f'--remote-debugging-port={cdp_port}' in cmdline:
                        proc.terminate()
                        terminated_pids.append(proc.info['pid'])
                        logger.info(f"Terminated Chrome process with PID {proc.info['pid']} (CDP-related)")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e: # Added ZombieProcess
                 # Log warnings for processes that can't be accessed or don't exist anymore
                 # logger.warning(f"Could not access or terminate process {proc.pid if hasattr(proc, 'pid') else 'N/A'}: {e}")
                 pass # Often these are expected, so just pass silently unless debugging needed
            except Exception as e:
                 logger.error(f"Unexpected error checking process {proc.pid if hasattr(proc, 'pid') else 'N/A'}: {e}")


        # Wait for processes to terminate
        gone, alive = psutil.wait_procs([psutil.Process(pid) for pid in terminated_pids], timeout=3)
        for p in alive:
             logger.warning(f"Process {p.pid} did not terminate, attempting kill.")
             try:
                 p.kill()
             except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                 logger.warning(f"Could not kill process {p.pid}: {e}")


        if terminated_pids:
             logger.info(f"Attempted termination of {len(terminated_pids)} Chrome processes.")
        else:
             logger.info("No Chrome processes found associated with CDP port {cdp_port} to terminate.")


    except Exception as e:
        logger.error(f"Error during Chrome process termination: {e}")


async def close_browser_resources(browser: CustomBrowser | None, browser_context: CustomBrowserContext | None):
    try:
        if browser_context:
            logger.info("Attempting to close browser context...")
            await browser_context.close()
            logger.info("Browser context closed successfully.")
        else:
            logger.info("Browser context already closed or not initialized.")
    except Exception as e:
        logger.error(f"Error closing browser context: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if browser:
                logger.info("Attempting to close browser...")
                await browser.close()
                logger.info("Browser closed successfully.")
            else:
                logger.info("Browser already closed or not initialized.")
        except Exception as e:
            logger.error(f"Error closing browser: {e}\n{traceback.format_exc()}")
        finally:
            # Always attempt termination after trying to close gracefully
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
    run_count: int = 1 # Pass run_count for logging and key cycling
):
    """Run a browser job with retry mechanism, proxies, and stealth techniques."""
    attempt = 1
    global_browser = None
    global_browser_context = None
    global_agent = None

    # Consider moving this list outside if it's static or loading from config
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
                    f"--window-size={current_window_w},{current_window_h}",
                    "--no-first-run",
                    "--no-default-browser-check",
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
                    max_steps=max_steps,
                    useOwnBrowser=True, # Important: Use the browser we set up
                    enable_emunium=False, # Control Emunium usage here if needed (already init'd)
                    customHistory=True
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

        except (ConnectionRefusedError, ConnectionResetError) as conn_err:
            error_msg = f"Connection Error on Run {run_count}, Attempt {attempt}: {conn_err}. Likely CDP issue or proxy failure.\n{traceback.format_exc()}"
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Error on Run {run_count}, Attempt {attempt}: {type(e).__name__} - {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            # Capture screenshot for debugging if possible
            if global_browser_context and not global_browser_context.is_closed():
                try:
                    page = await global_browser_context.get_current_page()
                    if page and not page.is_closed():
                        screenshot_path = os.path.abspath(f"./tmp/screenshot_error_run_{run_count}_attempt_{attempt}.png")
                        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                        await page.screenshot(path=screenshot_path, full_page=True)
                        logger.info(f"Error screenshot saved to {screenshot_path}")
                    else:
                        logger.warning("Could not save error screenshot: Page is closed or null.")
                except Exception as ss_err:
                    logger.error(f"Failed to take error screenshot: {ss_err}")

            # --- Retry Logic ---
            if attempt < max_attempts_per_task:
                attempt += 1
                logger.info(f"Attempt {attempt -1} failed. Closing resources before retry...")
                await close_browser_resources(global_browser, global_browser_context)
                global_browser = None # Ensure resources are reset
                global_browser_context = None
                global_agent = None
                wait_time = retry_delay + random.uniform(-5, 5) # Add jitter
                logger.info(f"Retrying task for Run {run_count} in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Max attempts ({max_attempts_per_task}) reached for Task Run {run_count}. Failing this run.")
                await close_browser_resources(global_browser, global_browser_context) # Ensure cleanup on final failure
                return None # Indicate failure for this run_count
        except Exception as outer_e:
             # Catch errors happening outside the main try-block (e.g., during playwright startup)
             logger.critical(f"Critical error during setup/retry logic for Run {run_count}, Attempt {attempt}: {outer_e}\n{traceback.format_exc()}")
             await close_browser_resources(global_browser, global_browser_context) # Ensure cleanup
             # Decide if you want to retry here or just fail the run
             if attempt < max_attempts_per_task:
                  attempt += 1
                  wait_time = retry_delay + random.uniform(-5, 5)
                  logger.info(f"Retrying task for Run {run_count} after critical error in {wait_time:.2f} seconds...")
                  await asyncio.sleep(wait_time)
             else:
                  logger.error(f"Max attempts ({max_attempts_per_task}) reached for Task Run {run_count} after critical error.")
                  return None


    logger.warning(f"Task Run {run_count} did not succeed within {max_attempts_per_task} attempts.")
    return None # Should not be reached if logic is correct, but acts as a safeguard


async def main_loop():
    """Main loop to run tasks with prompts modified by country/language based on run_count."""
    # --- Configuration ---
    PROMPT_FILE = "prompts/comments/gather_prompt4.json"
    MAX_RUNS = 5000  # Set total number of runs desired
    MAX_STEPS_PER_RUN = 200
    MAX_ACTIONS_PER_STEP = 3
    RETRY_DELAY_SECONDS = 30 # Increased base delay
    MAX_ATTEMPTS_PER_RUN = 3
    WAIT_BETWEEN_RUNS_SECONDS = 30 # Increased wait between distinct runs

    # Define region/language configurations based on run count ranges
    # (max_run_inclusive, country, language, text_to_type_pattern)
    REGION_CONFIG = [
        (100, "Germany", "German", "Suche nach deutschen Kommentaren Lauf {run_count}"),
        (200, "United Kingdom", "English", "Search for UK comments run {run_count}"),
        (300, "Italy", "Italian", "Cerca commenti italiani run {run_count}"),
        (400, "France", "French", "Recherche de commentaires français run {run_count}"),
        (500, "Spain", "Spanish", "Buscar comentarios españoles run {run_count}"),
        (600, "Australia", "English", "Search for Australian comments run {run_count}"),
        (700, "Austria", "German", "Suche nach österreichischen Kommentaren Lauf {run_count}"),
        (800, "Azerbaijan", "Azerbaijani", "Azərbaycanca şərhlər axtarın {run_count}"), # Simple translation
        (900, "Argentina", "Spanish", "Buscar comentarios argentinos run {run_count}"),
        (1000, "Armenia", "Armenian", "Որոնել հայերեն մեկնաբանություններ {run_count}"), # Simple translation
        (1100, "Belgium", "French/Dutch", "Search for Belgian comments run {run_count}"), # Use English as common ground or pick one
        (1200, "Bulgaria", "Bulgarian", "Търсене на български коментари {run_count}"), # Simple translation
        # Add more configurations as needed
    ]
    # Default if run_count exceeds defined ranges
    DEFAULT_CONFIG = ("Default", "English", "Generic search query run {run_count}")

    # --- Load Base Prompt ---
    base_task_goal, base_add_infos = load_json_prompt(file_path=PROMPT_FILE)
    if not base_task_goal:
        logger.error(f"Failed to load base task from '{PROMPT_FILE}'. Exiting.")
        return

    # --- Main Execution Loop ---
    run_count = 0 # Start from 0, will be incremented before first use
    while run_count < MAX_RUNS:
        run_count += 1
        logger.info(f"================ Starting Run {run_count}/{MAX_RUNS} ================")

        # --- Determine Region/Language/Text for this run ---
        selected_config = None
        for max_r, country, lang, pattern in REGION_CONFIG:
            if run_count <= max_r:
                selected_config = (country, lang, pattern)
                break

        if not selected_config:
            logger.warning(f"Run count {run_count} exceeds defined ranges. Using default config.")
            selected_config = DEFAULT_CONFIG

        country, language, text_pattern = selected_config
        text_to_type = text_pattern.format(run_count=run_count)

        # --- Construct the dynamic task prompt for the agent ---
        # Instruction: Go to search, type specific text WITHOUT advanced search, then do original task.
        initial_action_prompt = (
            f"1. Go to a standard search engine (like Google, DuckDuckGo, Bing, etc.).\n"
            f"2. Locate the main search input field.\n"
            f"3. VERY IMPORTANT: Type the following exact text into the search bar: '{text_to_type}'\n"
            f"4. Press Enter or click the primary search button.\n"
            f"5. CRITICAL: Do NOT use any 'Advanced Search' links, options, filters, or date range tools offered by the search engine for this initial search action. Just type the text and search directly.\n\n"
            f"After you have performed this search, proceed with the main objective:"
        )

        final_task_prompt = f"{initial_action_prompt}\n\n{base_task_goal}"

        # --- Construct dynamic additional info ---
        context_info = f"Run Context: Count={run_count}, Target Country={country}, Target Language={language}. You must type the specific text provided in the main task instructions."
        final_add_infos = f"{base_add_infos}\n\n{context_info}" if base_add_infos else context_info

        logger.info(f"Run {run_count}: Country={country}, Language={language}")
        logger.info(f"Run {run_count}: Text to type='{text_to_type}'")
        # logger.debug(f"Run {run_count}: Full Task Prompt:\n{final_task_prompt}") # Log full prompt only if needed (can be long)
        # logger.debug(f"Run {run_count}: Full Add Infos:\n{final_add_infos}")

        # --- Execute the browser job ---
        try:
            result = await run_browser_job(
                task=final_task_prompt,
                add_infos=final_add_infos,
                max_steps=MAX_STEPS_PER_RUN,
                max_actions_per_step=MAX_ACTIONS_PER_STEP,
                retry_delay=RETRY_DELAY_SECONDS,
                max_attempts_per_task=MAX_ATTEMPTS_PER_RUN,
                run_count=run_count, # Pass current run_count
                # Keep other parameters like use_vision, cdp_url etc. as needed
                use_vision=False,
                enable_emunium=True,
                keep_browser_open=False # Usually want to close between runs for clean state
            )

            if result:
                logger.info(f"Run {run_count} completed successfully. Result: {result}")
            else:
                logger.warning(f"Run {run_count} failed after all {MAX_ATTEMPTS_PER_RUN} attempts.")

            logger.info(f"Waiting {WAIT_BETWEEN_RUNS_SECONDS} seconds before next run...")
            await asyncio.sleep(WAIT_BETWEEN_RUNS_SECONDS)

        except Exception as e:
            logger.error(f"Critical unexpected error in main_loop during run {run_count}: {str(e)}\n{traceback.format_exc()}")
            logger.info(f"Waiting {WAIT_BETWEEN_RUNS_SECONDS} seconds before attempting next run...")
            # Make sure resources are cleaned up if an error bubbles up to here
            # This might be redundant if run_browser_job handles its cleanup, but safer
            terminate_chrome_process(cdp_port=9222)
            await asyncio.sleep(WAIT_BETWEEN_RUNS_SECONDS)


    logger.info(f"Completed {MAX_RUNS} runs.")


if __name__ == "__main__":
    # Ensure tmp directories exist
    os.makedirs("./tmp/traces", exist_ok=True)
    os.makedirs("./tmp/record_videos", exist_ok=True)
    os.makedirs("./tmp/agent_history", exist_ok=True)
    os.makedirs("./tmp/screenshots", exist_ok=True)
    os.makedirs("./tmp/agent_history_successful", exist_ok=True)
    os.makedirs("./tmp/screenshots_successful", exist_ok=True)

    try:
        logger.info("Starting main execution loop...")
        asyncio.run(main_loop())
        logger.info("Main execution loop finished.")
    except KeyboardInterrupt:
        logger.info("Program terminated by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Critical error running main_loop: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info("Performing final cleanup...")
        # Attempt final cleanup, although async resources might be tricky here
        terminate_chrome_process(cdp_port=9222)
        logger.info("Cleanup attempt finished. Exiting.")
