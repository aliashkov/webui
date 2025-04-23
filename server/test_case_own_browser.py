import asyncio
import os
import sys

import logging
import traceback
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

# Configure logging (matching webui.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_browser_job(
    task: str = (
        "1. Go to google.com. "
        "2. Move to the search bar, type 'hitz.me', and press Enter. "
        "3. Click the first search result to visit hitz.me. "
        "4. Find and click the 'Login' button. "
        "5. In the login form, enter email 'sagav74082@apklamp.com' and password 'test213', then submit the form. "
        "6. After successful login, perform the following actions: "
        "- Navigate to a music or video section and play a song or video. "
        "- Like a song, video, or post by clicking a 'Like' or heart button. "
        "- Leave a comment on a song, video, or post, typing 'Great track!' and submitting it. "
        "- Scroll down the page to load more content, then scroll back up. "
        "7. Ensure all actions are completed in sequence."
    ),
    cdp_url: str = "http://localhost:9222",
    window_w: int = 1280,
    window_h: int = 1025,
    max_steps: int = 10,  # Increased to match webui.py
    max_actions_per_step: int = 3,  # Increased to match webui.py
    use_vision: bool = False,
    enable_emunium: bool = True,
    keep_browser_open: bool = True
):
    """Run a browser job using an existing Chrome instance via CDP, adapted from webui.py's run_custom_agent."""
    async with async_playwright() as p:
        # Configure LLM (same as webui.py)
        llm = utils.get_llm_model(
            provider="google",
            model_name="gemini-2.0-flash",
            temperature=0.6,
            api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("GOOGLE_API_KEY environment variable not set")
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        global_browser = None
        global_browser_context = None
        global_agent = None

        try:
            # Initialize browser (from webui.py's run_custom_agent)
            extra_chromium_args = [f"--window-size={window_w},{window_h}"]
            cdp_url = os.getenv("CHROME_CDP", cdp_url)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]

            logger.info("Using CDP URL: %s", cdp_url)

            # Create CustomBrowser with CDP URL
            browser_config = BrowserConfig(
                headless=False,
                disable_security=True,
                cdp_url=cdp_url,
                chrome_instance_path=chrome_path,
                extra_chromium_args=extra_chromium_args
            )
            global_browser = CustomBrowser(config=browser_config)
            logger.info("CustomBrowser initialized with CDP URL: %s", cdp_url)

            # Create browser context
            global_browser_context = await global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path="./tmp/traces" if os.path.exists("./tmp/traces") else None,
                    save_recording_path="./tmp/record_videos" if os.path.exists("./tmp/record_videos") else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h)
                )
            )
            logger.info("Browser Context created: %s", global_browser_context)

            # Initialize controller and page
            controller = CustomController()
            page = await global_browser_context.get_current_page()
            emunium = EmuniumPlaywright(page)
            logger.info("Page initialized: %s", page)

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
            logger.info("Final Result: %s", history.final_result())

            # Save history (as in webui.py)
            history_file = os.path.join("./tmp/agent_history", f"{global_agent.state.agent_id}.json")
            os.makedirs("./tmp/agent_history", exist_ok=True)
            global_agent.save_history(history_file)
            logger.info("Agent history saved to %s", history_file)

            return history.final_result()

        except Exception as e:
            logger.error("Error running browser job: %s\n%s", str(e), traceback.format_exc())
            raise
        finally:
            # Cleanup (improved to match webui.py and avoid NoneType errors)
            global_agent = None
            if not keep_browser_open:
                try:
                    if global_browser_context and global_browser_context:  # Check if context is valid
                        await global_browser_context.close()
                        logger.info("Browser context closed")
                    global_browser_context = None
                except Exception as e:
                    logger.warning("Failed to close browser context: %s", str(e))
                
                try:
                    if global_browser:  # Check if browser is valid
                        await global_browser.close()
                        logger.info("Browser closed")
                    global_browser = None
                except Exception as e:
                    logger.warning("Failed to close browser: %s", str(e))

if __name__ == "__main__":
    asyncio.run(run_browser_job())