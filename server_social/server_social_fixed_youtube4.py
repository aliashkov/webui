import asyncio
import os
import shutil
import sys
import logging
import traceback
import time
import json
import random
from typing import Optional # Added for type hinting

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
    "TASK_TIME_LIMIT_SECONDS": 600,  # 20 minutes: Time limit for a single agent task execution
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

async def close_browser_resources(browser: Optional[CustomBrowser], browser_context: Optional[CustomBrowserContext]):
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
    task_time_limit_seconds: int = CONFIG["TASK_TIME_LIMIT_SECONDS"] # New parameter
):
    """Run a browser job with retry mechanism, proxies, stealth techniques, and time limit."""
    attempt = 1
    global_browser = None
    global_browser_context = None
    global_agent = None
    
    while attempt <= CONFIG["MAX_ATTEMPTS_PER_TASK"]:
        logger.info(f"Attempt {attempt} of {CONFIG['MAX_ATTEMPTS_PER_TASK']} for task. Time limit: {task_time_limit_seconds}s.")
        async with async_playwright() as p:
            try:
                # Step 1: Ensure Chrome is terminated before starting
                terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])
                await asyncio.sleep(2)

                # Step 2: Configure LLM with cycling API keys
                key_index = run_count % 7
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
                    provider="google", model_name="gemini-2.0-flash",
                    temperature=0.6, api_key=api_key
                )
                logger.info(f"Using {api_key_name} for run {run_count}")

                # Step 3: Initialize browser with proxy and stealth
                current_window_w = random.randint(window_w, window_w + 200)
                current_window_h = random.randint(window_h, window_h + 200)
                extra_chromium_args = [
                    f"--window-size={current_window_w},{current_window_h}",
                ]
                
                proxy_server = random.choice(CONFIG["PROXY_LIST"])
                proxy_details = { "server": f"http://{proxy_server}", "username": "", "password": "" }
                logger.info(f"Using proxy: {proxy_server}")
                
                cdp_url_env = os.getenv("CHROME_CDP", cdp_url)
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "": chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data == "": chrome_user_data = None
                if chrome_user_data:
                     extra_chromium_args.append(f"--user-data-dir={chrome_user_data}")

                browser_config = BrowserConfig(
                    headless=False, disable_security=True, cdp_url=cdp_url_env,
                    chrome_instance_path=chrome_path, extra_chromium_args=extra_chromium_args,
                    proxy=proxy_details
                )
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser initialized with CDP URL: {cdp_url_env}")

                # Step 4: Create browser context
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp/traces") else None,
                        save_recording_path="./tmp/record_videos" if os.path.exists("./tmp/record_videos") else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(width=current_window_w, height=current_window_h)
                    )
                )
                logger.info(f"Browser Context created: {global_browser_context}")

                # Step 5: Initialize controller and page
                controller = CustomController()
                page = await global_browser_context.get_current_page()
                if CONFIG["ENABLE_EMUNIUM"]: # Conditionally initialize Emunium
                    emunium = EmuniumPlaywright(page)
                    logger.info("EmuniumPlaywright initialized.")
                else:
                    logger.info("EmuniumPlaywright disabled by config.")
                logger.info(f"Page initialized: {page}")

                await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(random.uniform(1, 3))
                
                # Step 6: Create and run agent
                global_agent = CustomAgent(
                    task=task, add_infos=add_infos, use_vision=CONFIG["USE_VISION"],
                    llm=llm, browser=global_browser, browser_context=global_browser_context,
                    controller=controller, system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=CONFIG["MAX_ACTIONS_PER_STEP"],
                    tool_calling_method="auto", max_input_tokens=128000, generate_gif=True
                )

                # Calculate deadline for agent's run
                agent_start_time = time.time()
                agent_deadline = agent_start_time + task_time_limit_seconds
                
                logger.info(f"Starting agent. Max steps: {CONFIG['MAX_STEPS']}, Time limit: {task_time_limit_seconds}s.")

                # IMPORTANT: You need to modify your CustomAgent.run method to accept and use the 'deadline' parameter.
                # See the comment block after this script for an example.
                history = await global_agent.run(
                    max_steps=CONFIG["MAX_STEPS"],
                    deadline=agent_deadline, # Pass the calculated deadline
                    useOwnBrowser=CONFIG["USE_OWN_BROWSER"],
                    enable_emunium=CONFIG["ENABLE_EMUNIUM"],
                    customHistory=True,
                    enableEnter=CONFIG["ENABLE_ENTER"],
                    enableClick=CONFIG["ENABLE_CLICK"]
                )
                
                # Check if task was stopped due to time limit (if agent doesn't set specific status)
                # This is a fallback check; ideally, agent.run() and history reflect timeout.
                if time.time() >= agent_deadline and not history.is_successful():
                    if history.final_result() and "time" in history.final_result().lower() and "out" in history.final_result().lower():
                        logger.warning(f"Task for agent {global_agent.state.agent_id} confirmed timed out by agent's final result.")
                    else:
                        logger.warning(f"Task for agent {global_agent.state.agent_id} may have hit the time limit of {task_time_limit_seconds}s and did not complete successfully.")
                        # If your agent doesn't update its history/final_result on timeout,
                        # you might want to force a failure status here or ensure history reflects it.
                        # For now, relying on history.is_successful() and agent's own timeout handling.

                logger.info(f"Task completed. Final Result: {history.final_result()}")

                # Step 7: Determine success status and save history/screenshots
                success = history.is_successful()
                logger.info(f"Task success status: {success}")

                base_history_dir = "./tmp/agent_history"
                base_screenshot_dir = "./tmp/screenshots"
                successful_history_dir = "./tmp/agent_history_successful"
                successful_screenshot_dir = "./tmp/screenshots_successful"

                os.makedirs(base_history_dir, exist_ok=True)
                os.makedirs(base_screenshot_dir, exist_ok=True)
                if success:
                    os.makedirs(successful_history_dir, exist_ok=True)
                    os.makedirs(successful_screenshot_dir, exist_ok=True)

                history_file = os.path.join(base_history_dir, f"{global_agent.state.agent_id}.json")
                successful_history_file = os.path.join(successful_history_dir, f"{global_agent.state.agent_id}.json") if success else None
                
                run_screenshot_target_dir = os.path.join(base_screenshot_dir, f"{global_agent.state.agent_id}_run_{run_count}")
                successful_run_screenshot_target_dir = os.path.join(successful_screenshot_dir, f"{global_agent.state.agent_id}_run_{run_count}") if success else None
                
                os.makedirs(run_screenshot_target_dir, exist_ok=True)
                if success and successful_run_screenshot_target_dir:
                    os.makedirs(successful_run_screenshot_dir, exist_ok=True)

                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")
                if success and successful_history_file:
                    # shutil.copy(history_file, successful_history_file) # Copy if save_history doesn't take a dynamic path
                    global_agent.save_history(successful_history_file) # Or save again if that's how it works
                    logger.info(f"Agent history also saved to {successful_history_file}")

                # Adjust screenshot source path based on your CustomAgent's saving behavior
                source_screenshots_path = os.path.join("./tmp", "screenshots", global_agent.state.agent_id) 
                if not os.path.exists(source_screenshots_path) and hasattr(global_agent, 'run_number'): 
                    source_screenshots_path = os.path.join("./tmp", "screenshots", f"run_{global_agent.run_number}")

                if os.path.exists(source_screenshots_path):
                    os.makedirs(run_screenshot_target_dir, exist_ok=True)
                    if success and successful_run_screenshot_target_dir:
                         os.makedirs(successful_run_screenshot_target_dir, exist_ok=True)

                    for filename in os.listdir(source_screenshots_path):
                        src_path = os.path.join(source_screenshots_path, filename)
                        dst_path_main = os.path.join(run_screenshot_target_dir, filename)
                        try:
                            shutil.copy(src_path, dst_path_main)
                            logger.info(f"Screenshot copied to {dst_path_main}")
                            if success and successful_run_screenshot_target_dir:
                                dst_path_successful = os.path.join(successful_run_screenshot_target_dir, filename)
                                shutil.copy(src_path, dst_path_successful)
                                logger.info(f"Screenshot also copied to {dst_path_successful}")
                        except Exception as e_mv:
                            logger.error(f"Error copying screenshot {src_path}: {e_mv}")
                else:
                    logger.warning(f"Screenshot source directory {source_screenshots_path} not found for agent {global_agent.state.agent_id}.")

                return history.final_result()

            except Exception as e:
                error_msg = f"Error on attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                if global_browser_context:
                    try:
                        page = await global_browser_context.get_current_page()
                        if page and not page.is_closed():
                            screenshot_path = f"./tmp/error_screenshot_run_{run_count}_attempt_{attempt}.png"
                            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                            await page.screenshot(path=screenshot_path)
                            logger.info(f"Error screenshot saved to {screenshot_path}")
                        elif page and page.is_closed():
                            logger.warning("Page was closed, cannot take error screenshot.")
                        else:
                            logger.warning("Page object not available, cannot take error screenshot.")
                    except Exception as se:
                        logger.error(f"Failed to take error screenshot: {se}")

                if attempt < CONFIG["MAX_ATTEMPTS_PER_TASK"]:
                    logger.info(f"Retrying in {CONFIG['RETRY_DELAY']} seconds...")
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser, global_browser_context, global_agent = None, None, None
                    await asyncio.sleep(CONFIG["RETRY_DELAY"])
                    attempt += 1
                else:
                    logger.error(f"Max attempts ({CONFIG['MAX_ATTEMPTS_PER_TASK']}) reached for task.")
                    await close_browser_resources(global_browser, global_browser_context) # Ensure cleanup on final failure
                    return None 
            finally:
                # Close resources if not keeping browser open, or if it's the last attempt (success or fail)
                if not CONFIG["KEEP_BROWSER_OPEN"] or attempt > CONFIG["MAX_ATTEMPTS_PER_TASK"]:
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser, global_browser_context, global_agent = None, None, None
        
    logger.warning(f"Task failed after {CONFIG['MAX_ATTEMPTS_PER_TASK']} attempts.")
    return None


async def main_loop():
    task, add_infos = load_json_prompt(file_path="prompts/youtube_multi_promo_prompt3.json")
    if not task:
        logger.error("Failed to load task from JSON prompt file. Exiting.")
        return

    run_count = 0
    max_runs = CONFIG["MAX_RUNS"] if CONFIG["MAX_RUNS"] is not None else float('inf')


    while run_count < max_runs:
        run_count += 1
        logger.info(f"Starting run {run_count} of {'unlimited' if max_runs == float('inf') else max_runs}")

        try:
            result = await run_browser_job(
                task=task,
                add_infos=add_infos,
                run_count=run_count,
                # You can override the default time limit from CONFIG here if needed for specific tasks
                # task_time_limit_seconds=specific_time_limit_for_this_task 
            )
            if result:
                logger.info(f"Run {run_count} completed. Result: {result}")
            else:
                logger.warning(f"Run {run_count} failed or returned no result after all attempts.")
            
            if run_count < max_runs: # Avoid delay if it's the last run
                logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before next run (if applicable)...")
                await asyncio.sleep(CONFIG["RETRY_DELAY"])

        except Exception as e:
            logger.error(f"Unexpected error in main_loop during run {run_count}: {str(e)}\n{traceback.format_exc()}")
            if run_count < max_runs: # Avoid delay if it's the last run
                logger.info(f"Waiting {CONFIG['RETRY_DELAY']} seconds before attempting next run...")
                await asyncio.sleep(CONFIG["RETRY_DELAY"])

if __name__ == "__main__":
    for dirname in ["./tmp/agent_history", "./tmp/screenshots", 
                    "./tmp/agent_history_successful", "./tmp/screenshots_successful",
                    "./tmp/traces", "./tmp/record_videos"]:
        os.makedirs(dirname, exist_ok=True)
    
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main application: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info("Main loop finished or terminated. Performing final cleanup.")
        # terminate_chrome_process is called within run_browser_job's finally block,
        # but an extra call here might be a good safety net if main_loop exits unexpectedly.
        terminate_chrome_process(cdp_port=CONFIG["CDP_PORT"])