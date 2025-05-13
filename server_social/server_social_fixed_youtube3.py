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
    "RETRY_ATTEMPT_DELAY": 25,  # Delay between retry attempts for a failing task in seconds
    "WORK_SESSION_BREAK_DURATION_SECONDS": 10 * 60, # 10 minutes break after a work session
    "ENABLE_EMUNIUM": True,
    "MAX_STEPS_PER_WORK_SESSION": 180, # Max steps for the agent in one ~30-min work session
    "MAX_ACTIONS_PER_STEP": 3,
    "MAX_ATTEMPTS_PER_WORK_SESSION": 3, # Max retry attempts for a single work session if it fails
    "ENABLE_ENTER": False,
    "CDP_PORT": 9222,
    "WINDOW_WIDTH": 1500,
    "WINDOW_HEIGHT": 1025,
    "KEEP_BROWSER_OPEN_DURING_SESSION": False, # If true, browser stays open until script ends. If false, closes after each work session.
                                            # For "close tab during break", this should be False.
    "USE_VISION": False,
    "MAX_WORK_SESSIONS": 5000, # Maximum number of work sessions in main loop
    "USE_OWN_BROWSER": True,
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

def load_json_prompt(file_path: str) -> tuple[str, str]: # Removed default
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
                if proc.info['name'] and proc.info['name'].lower() == 'chrome.exe': # Added check for proc.info['name']
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and f'--remote-debugging-port={cdp_port}' in cmdline:
                        proc.terminate()
                        terminated_pids.append(proc.info['pid'])
                        logger.info(f"Terminated Chrome process with PID {proc.info['pid']} (CDP-related)")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not access process {proc.info.get('pid', 'N/A')}: {e}") # Safer get
            except Exception as e_proc: # Catch other potential errors like NoneType cmdline
                logger.warning(f"Error processing process {proc.info.get('pid', 'N/A')}: {e_proc}")


        time.sleep(1) # Give processes time to terminate
        if not terminated_pids:
            logger.info(f"No Chrome processes found listening on CDP port {cdp_port} to terminate.")
        else:
            logger.info(f"Terminated {len(terminated_pids)} Chrome processes related to CDP port {cdp_port}.")
    except Exception as e:
        logger.error(f"Error during Chrome process termination scan: {e}")


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
            # else: # Not really needed, browser might not be initialized on early failure
            #     logger.info("Browser already closed or not initialized.")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
        finally:
            # Terminate Chrome processes more reliably after attempting to close browser via Playwright
            terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])


async def run_browser_job(
    task: str,
    add_infos: str = "",
    default_cdp_url: str = "http://localhost:9222", # Renamed to avoid clash
    window_w: int = CONFIG["WINDOW_WIDTH"],
    window_h: int = CONFIG["WINDOW_HEIGHT"],
    session_count: int = 1, # Renamed from run_count for clarity
):
    """Run a browser job (one work session) with retry mechanism, proxies, and stealth techniques."""
    attempt = 1
    # These are now local to each attempt, ensuring no stale state
    # global_browser = None 
    # global_browser_context = None
    # global_agent = None
    
    while attempt <= CONFIG["MAX_ATTEMPTS_PER_WORK_SESSION"]:
        logger.info(f"Work Session {session_count}, Attempt {attempt} of {CONFIG['MAX_ATTEMPTS_PER_WORK_SESSION']}")
        
        current_browser = None
        current_browser_context = None
        # current_agent = None # Agent is created later

        async with async_playwright() as p:
            try:
                # Step 1: Ensure Chrome is terminated before starting
                terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])
                await asyncio.sleep(2) # Give OS time to release port

                # Step 2: Configure LLM with cycling API keys
                key_index = (session_count -1) % 7 # session_count starts at 1
                api_key_map = {
                    0: "GOOGLE_API_KEY", 1: "GOOGLE_API_KEY2", 2: "GOOGLE_API_KEY3",
                    3: "GOOGLE_API_KEY4", 4: "GOOGLE_API_KEY5", 5: "GOOGLE_API_KEY6",
                    6: "GOOGLE_API_KEY7"
                }
                api_key_name = api_key_map[key_index]
                api_key = os.getenv(api_key_name, "")
                if not api_key:
                    logger.error(f"{api_key_name} environment variable not set")
                    raise ValueError(f"{api_key_name} environment variable not set")
                
                llm = utils.get_llm_model(
                    provider="google", model_name="gemini-1.5-flash", # Updated model
                    temperature=0.6, api_key=api_key
                )
                logger.info(f"Using {api_key_name} for Work Session {session_count}")

                # Step 3: Initialize browser with proxy and stealth
                current_window_w = random.randint(window_w, window_w + 200)
                current_window_h = random.randint(window_h, window_h + 200)
                extra_chromium_args = [
                    f"--window-size={current_window_w},{current_window_h}",
                ]
                
                proxy_server = random.choice(CONFIG["PROXY_LIST"]) if CONFIG["PROXY_LIST"] else None
                proxy_config = None
                if proxy_server:
                    proxy_config = { "server": f"http://{proxy_server}" } # Assuming no auth for these proxies
                    logger.info(f"Using proxy: {proxy_server}")
                else:
                    logger.info("No proxies configured or list is empty, running without proxy.")

                
                # Use environment variable for CDP URL if set, otherwise use default_cdp_url
                cdp_url_to_use = os.getenv("CHROME_CDP", default_cdp_url)
                chrome_path = os.getenv("CHROME_PATH") or None # Handles empty string
                chrome_user_data_dir = os.getenv("CHROME_USER_DATA") or None # Handles empty string
                
                if chrome_user_data_dir:
                    extra_chromium_args.append(f"--user-data-dir={chrome_user_data_dir}")
                    logger.info(f"Using Chrome user data directory: {chrome_user_data_dir}")


                browser_config = BrowserConfig(
                    headless=False, # For interactive tasks, usually False
                    disable_security=True,
                    cdp_url=cdp_url_to_use,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                    proxy=proxy_config
                )
                current_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser initialized. CDP URL: {cdp_url_to_use}, Headless: {browser_config.headless}")

                # Step 4: Create browser context
                current_browser_context = await current_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path=os.path.join("./tmp/traces", f"session_{session_count}_attempt_{attempt}") if os.path.exists("./tmp/traces") else None,
                        save_recording_path=os.path.join("./tmp/record_videos", f"session_{session_count}_attempt_{attempt}.webm") if os.path.exists("./tmp/record_videos") else None,
                        no_viewport=False, # Explicitly set viewport
                        browser_window_size=BrowserContextWindowSize(width=current_window_w, height=current_window_h)
                    )
                )
                logger.info(f"Browser Context created with viewport {current_window_w}x{current_window_h}")

                # Step 5: Initialize controller and page
                controller = CustomController()
                page = await current_browser_context.get_current_page()
                if not page: # Robustness check
                    logger.error("Failed to get current page from context.")
                    raise Exception("Failed to get current page from context.")
                
                # Emunium is initialized here if enabled in CONFIG
                # emunium = EmuniumPlaywright(page) # This is used by agent if enable_emunium=True
                logger.info(f"Page initialized: {page.url if page else 'No Page'}")

                # Simulate human-like startup behavior
                await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                await page.evaluate("() => { window.scrollTo(0, Math.floor(document.body.scrollHeight * Math.random())); }") # Random scroll
                await asyncio.sleep(random.uniform(1, 3))
                
                # Step 6: Create and run agent
                agent = CustomAgent( # Renamed from global_agent
                    task=task,
                    add_infos=add_infos,
                    use_vision=CONFIG["USE_VISION"],
                    llm=llm,
                    browser=current_browser,
                    browser_context=current_browser_context,
                    controller=controller,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=CONFIG["MAX_ACTIONS_PER_STEP"],
                    tool_calling_method="auto",
                    max_input_tokens=1000000, # Gemini 1.5 Flash can handle 1M
                    generate_gif=True # Control this via config if needed
                )
                # current_agent = agent # Assign to current_agent

                history = await agent.run(
                    max_steps=CONFIG["MAX_STEPS_PER_WORK_SESSION"],
                    useOwnBrowser=CONFIG["USE_OWN_BROWSER"],
                    enable_emunium=CONFIG["ENABLE_EMUNIUM"],
                    customHistory=True,
                    enableEnter=CONFIG["ENABLE_ENTER"],
                    enableClick=CONFIG["ENABLE_CLICK"]
                )
                logger.info(f"Work Session {session_count} (Attempt {attempt}) completed. Final Result: {history.final_result()}")

                # Step 7: Determine success status and save history/screenshots
                success = history.is_successful()
                logger.info(f"Work Session success status: {success}")

                # Define base directories
                base_history_dir = "./tmp/agent_history"
                base_screenshot_dir = "./tmp/screenshots" # Agent saves screenshots to ./tmp/screenshots/{agent_id}/
                
                # Specific directories for this session's artifacts
                session_history_dir = os.path.join(base_history_dir, f"session_{session_count}")
                session_screenshot_dir = os.path.join(base_screenshot_dir, f"session_{session_count}")
                
                # Suffix for successful runs
                success_suffix = "_successful" if success else ""
                
                final_history_dir = os.path.join(base_history_dir, f"session_{session_count}{success_suffix}")
                final_screenshot_dir = os.path.join(base_screenshot_dir, f"session_{session_count}{success_suffix}")

                os.makedirs(final_history_dir, exist_ok=True)
                os.makedirs(final_screenshot_dir, exist_ok=True) # For organizing agent's own screenshots

                # Save history
                history_file_name = f"{agent.state.agent_id}_attempt_{attempt}.json"
                history_file_path = os.path.join(final_history_dir, history_file_name)
                agent.save_history(history_file_path)
                logger.info(f"Agent history saved to {history_file_path}")

                # Screenshots are typically saved by the agent into a folder like ./tmp/screenshots/{agent.state.agent_id}/
                # We will move/copy them to the session-specific folder.
                agent_default_screenshot_path = os.path.join("./tmp/screenshots", agent.state.agent_id)

                if os.path.exists(agent_default_screenshot_path):
                    # Create a sub-directory for this attempt's screenshots within the session folder
                    attempt_screenshot_target_dir = os.path.join(final_screenshot_dir, f"attempt_{attempt}_{agent.state.agent_id}")
                    os.makedirs(attempt_screenshot_target_dir, exist_ok=True)
                    
                    for filename in os.listdir(agent_default_screenshot_path):
                        src_path = os.path.join(agent_default_screenshot_path, filename)
                        dst_path = os.path.join(attempt_screenshot_target_dir, filename)
                        try:
                            shutil.copy(src_path, dst_path) # Copy to keep original agent run if needed for other analysis
                            # If you want to move and then clean up the agent_id folder:
                            # shutil.move(src_path, dst_path)
                        except Exception as e_mv:
                            logger.error(f"Error copying screenshot {src_path} to {dst_path}: {e_mv}")
                    logger.info(f"Screenshots from {agent_default_screenshot_path} copied to {attempt_screenshot_target_dir}")
                    # Optionally, clean up the agent_default_screenshot_path if it's truly temporary for this run
                    # shutil.rmtree(agent_default_screenshot_path, ignore_errors=True)

                else:
                    logger.warning(f"Agent's default screenshot directory {agent_default_screenshot_path} not found.")
                
                # If successful, break from retry loop
                if success:
                    return history.final_result()
                else: # If not successful, but agent completed (e.g. max_steps reached without meeting goal)
                    logger.warning(f"Work session {session_count} attempt {attempt} did not achieve success criteria.")
                    # Continue to next attempt if any, or fail the work session.
                    # No explicit return here, will fall through to retry logic or finally block.

            except Exception as e:
                error_msg = f"Error on Work Session {session_count}, Attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                if current_browser_context: # Check if context was initialized
                    try:
                        page = await current_browser_context.get_current_page() # Re-fetch, might have changed
                        if page and not page.is_closed():
                            screenshot_path = os.path.join("./tmp", f"error_session_{session_count}_attempt_{attempt}.png")
                            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                            await page.screenshot(path=screenshot_path)
                            logger.info(f"Error screenshot saved to {screenshot_path}")
                        elif page and page.is_closed():
                            logger.warning("Page was closed, cannot take error screenshot.")
                        else:
                            logger.warning("Page object not available for error screenshot.")
                    except Exception as se:
                        logger.error(f"Failed to take error screenshot: {se}")
                # This attempt failed, loop will handle retry if attempt < MAX_ATTEMPTS_PER_WORK_SESSION
            
            finally:
                # Always close resources for this attempt, unless configured to keep open AND it's not the last attempt
                # The KEEP_BROWSER_OPEN_DURING_SESSION is for the entire script run, not per attempt.
                # For the work/break cycle, we want to close the browser after each session if KEEP_BROWSER_OPEN_DURING_SESSION is False.
                # Here, we close after each *attempt* to ensure a clean slate for retries.
                await close_browser_resources(current_browser, current_browser_context)
                current_browser = None 
                current_browser_context = None
                # current_agent = None # Agent is re-created

        # After playwright block, if we are going to retry
        if attempt < CONFIG["MAX_ATTEMPTS_PER_WORK_SESSION"]:
            logger.info(f"Retrying work session {session_count} in {CONFIG['RETRY_ATTEMPT_DELAY']} seconds...")
            await asyncio.sleep(CONFIG["RETRY_ATTEMPT_DELAY"])
        
        attempt += 1 # Increment attempt number

    logger.error(f"Work Session {session_count} failed after {CONFIG['MAX_ATTEMPTS_PER_WORK_SESSION']} attempts.")
    return None # All attempts for this work session failed


async def main_loop():
    """Main loop to run work sessions with breaks."""
    prompt_file_path = "prompts/youtube_promotion_prompt6.json" # New prompt file
    task, add_infos = load_json_prompt(file_path=prompt_file_path)
    if not task:
        logger.error(f"Failed to load task from '{prompt_file_path}'. Exiting.")
        return

    session_count = 0
    max_sessions = CONFIG["MAX_WORK_SESSIONS"]

    while max_sessions is None or session_count < max_sessions:
        session_count += 1
        logger.info(f"--- Starting Work Session {session_count} of {max_sessions if max_sessions else 'unlimited'} ---")

        try:
            result = await run_browser_job(
                task=task,
                add_infos=add_infos,
                session_count=session_count, # Pass current session number
            )
            if result:
                logger.info(f"Work Session {session_count} completed. Result: {result}")
            else:
                logger.warning(f"Work Session {session_count} failed or returned no result after all attempts.")
            
            # Break after the work session, regardless of its success, before starting the next one (if any)
            if max_sessions is None or session_count < max_sessions:
                logger.info(f"Work Session {session_count} finished. Taking a {CONFIG['WORK_SESSION_BREAK_DURATION_SECONDS']}s break...")
                await asyncio.sleep(CONFIG["WORK_SESSION_BREAK_DURATION_SECONDS"])
            else:
                logger.info(f"All {max_sessions} work sessions completed.")

        except Exception as e:
            logger.error(f"Unexpected error in main_loop during Work Session {session_count}: {str(e)}\n{traceback.format_exc()}")
            if max_sessions is None or session_count < max_sessions:
                logger.info(f"Attempting to recover, taking a {CONFIG['WORK_SESSION_BREAK_DURATION_SECONDS']}s break before next session...")
                await asyncio.sleep(CONFIG["WORK_SESSION_BREAK_DURATION_SECONDS"])


if __name__ == "__main__":
    # Create tmp directories if they don't exist
    os.makedirs("./tmp/agent_history", exist_ok=True)
    os.makedirs("./tmp/screenshots", exist_ok=True)
    # Removed _successful specific dirs here as they are created dynamically now
    os.makedirs("./tmp/traces", exist_ok=True)
    os.makedirs("./tmp/record_videos", exist_ok=True)
    
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main application: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info("Main loop finished or terminated. Performing final cleanup.")
        # Final cleanup, especially if KEEP_BROWSER_OPEN_DURING_SESSION was true and loop exited unexpectedly.
        # However, close_browser_resources calls terminate_chrome_process internally.
        # A standalone call here ensures termination if loop breaks before run_browser_job's finally.
        terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])
        logger.info("Exiting application.")