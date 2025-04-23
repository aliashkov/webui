import asyncio
import os
import sys
import logging
import traceback
import time
from playwright.async_api import async_playwright
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler('browser_job.log', mode='a')  # Log to file for persistence
    ]
)
logger = logging.getLogger(__name__)

def terminate_chrome_process(cdp_port=9222):
    for proc in psutil.process_iter(['name', 'cmdline']):
        if proc.info['name'] == 'chrome' and f'--remote-debugging-port={cdp_port}' in proc.info['cmdline']:
            proc.terminate()
            logger.info(f"Terminated Chrome process with PID {proc.pid}")

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
            # Terminate external Chrome process if using CDP
            terminate_chrome_process(cdp_port=9222)
        except Exception as e:
            logger.error(f"Error closing browser: {e}")

async def run_browser_job(
    task: str = (
        "type 'hitz.me,' click search. After that click to Login button and you need to type this 'sagav74082@apklamp.com test213'. After successfully login imitate different actions (watch music, like somewhere, leave comments and etc.). Also don't forget to scroll"
    ),
    cdp_url: str = "http://localhost:9222",
    window_w: int = 1280,
    window_h: int = 1025,
    max_steps: int = 2,
    max_actions_per_step: int = 3,
    use_vision: bool = False,
    enable_emunium: bool = True,
    keep_browser_open: bool = False,
    retry_delay: int = 25,  # Seconds to wait before retrying
    max_attempts_per_task: int = 3  # Maximum attempts for a single task
):
    """Run a browser job with retry mechanism and proper cleanup."""
    attempt = 1
    global_browser = None
    global_browser_context = None
    global_agent = None

    while attempt <= max_attempts_per_task:
        logger.info(f"Attempt {attempt} of {max_attempts_per_task} for task")
        async with async_playwright() as p:
            try:
                # Configure LLM
                llm = utils.get_llm_model(
                    provider="google",
                    model_name="gemini-2.0-flash",
                    temperature=0.6,
                    api_key=os.getenv("GOOGLE_API_KEY", "")
                )
                if not os.getenv("GOOGLE_API_KEY"):
                    logger.error("GOOGLE_API_KEY environment variable not set")
                    raise ValueError("GOOGLE_API_KEY environment variable not set")

                # Initialize browser
                extra_chromium_args = [f"--window-size={window_w},{window_h}"]
                cdp_url = os.getenv("CHROME_CDP", cdp_url)
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "":
                    chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data:
                    extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]

                logger.info(f"Using CDP URL: {cdp_url}")

                browser_config = BrowserConfig(
                    headless=False,
                    disable_security=True,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args
                )
                global_browser = CustomBrowser(config=browser_config)
                logger.info(f"CustomBrowser initialized with CDP URL: {cdp_url}")

                # Create browser context
                global_browser_context = await global_browser.new_context(
                    config=BrowserContextConfig(
                        trace_path="./tmp/traces" if os.path.exists("./tmp/traces") else None,
                        save_recording_path="./tmp/record_videos" if os.path.exists("./tmp/record_videos") else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h)
                    )
                )
                logger.info(f"Browser Context created: {global_browser_context}")

                # Initialize controller and page
                controller = CustomController()
                page = await global_browser_context.get_current_page()
                emunium = EmuniumPlaywright(page)
                logger.info(f"Page initialized: {page}")

                # Create and run agent
                global_agent = CustomAgent(
                    task=task,
                    add_infos="",
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
                    browserContext=global_browser_context,
                    useOwnBrowser=True,
                    enable_emunium=enable_emunium
                )
                logger.info(f"Task completed successfully. Final Result: {history.final_result()}")

                # Save history
                history_file = os.path.join("./tmp/agent_history", f"{global_agent.state.agent_id}.json")
                os.makedirs("./tmp/agent_history", exist_ok=True)
                global_agent.save_history(history_file)
                logger.info(f"Agent history saved to {history_file}")

                return history.final_result()  # Return result on success

            except Exception as e:
                error_msg = f"Error on attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                if attempt < max_attempts_per_task:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None
                    await asyncio.sleep(retry_delay)
                    attempt += 1
                else:
                    logger.error(f"Max attempts ({max_attempts_per_task}) reached for task. Moving to next task or stopping.")
                    return None  # Return None to indicate failure
            finally:
                if not keep_browser_open:
                    await close_browser_resources(global_browser, global_browser_context)
                    global_browser = None
                    global_browser_context = None
                    global_agent = None

    return None  # Return None if all attempts fail

async def main_loop():
    """Main loop to keep running tasks indefinitely."""
    task = (
        "type 'hitz.me,' click search. After that click to Login button and you need to type this 'sagav74082@apklamp.com test213'. After successfully login imitate different actions (watch music, like somewhere, leave comments and etc.). Also don't forget to scroll"
    )
    run_count = 0
    max_runs = None

    while max_runs is None or run_count < max_runs:
        run_count += 1
        logger.info(f"Starting run {run_count}")
        try:
            result = await run_browser_job(
                task=task,
                max_steps=2,
                max_actions_per_step=3,
                retry_delay=25,
                max_attempts_per_task=3
            )
            if result:
                logger.info(f"Run {run_count} completed successfully with result: {result}")
            else:
                logger.warning(f"Run {run_count} failed after all attempts.")
            logger.info(f"Waiting 25 seconds before next run...")
            await asyncio.sleep(25)
        except Exception as e:
            logger.error(f"Unexpected error in run {run_count}: {str(e)}\n{traceback.format_exc()}")
            logger.info(f"Waiting 25 seconds before retrying...")
            await asyncio.sleep(25)  # Delay before retrying on unexpected error

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main loop: {str(e)}\n{traceback.format_exc()}")