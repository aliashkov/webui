import asyncio
import sys
import os
from browser_use import BrowserConfig
from playwright.async_api import async_playwright

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.browser.custom_browser import CustomBrowser
from src.agent.custom_agent import CustomAgent
from src.controller.custom_controller import CustomController
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from src.utils import utils

async def test_custom_context():
    async with async_playwright() as p:
        llm = utils.get_llm_model(
            provider="google",
            model_name="gemini-2.0-flash",
            temperature=0.6,
            api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        browser = CustomBrowser(config=BrowserConfig(
                    headless=False
                ))
        context = await browser.new_context(config=BrowserContextConfig(
            browser_window_size=BrowserContextWindowSize(width=1280, height=1100)
        ))
        
        
        
        controller = CustomController()
        page = await context.get_current_page()
        print( page)
        await page.goto("https://duckduckgo.com/")
        await context.move_to_element('[class="searchbox_input__rnFzM"]')
        
        print("Context", context)
        print("Browser", browser)
        agent = CustomAgent(
            task="Go to google.com, move to the search bar, type 'OpenAI', move to the search button, and click it",
            llm=llm,
            browser=browser,
            browser_context=context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            use_vision=False,
            max_actions_per_step=3  # Increased to allow cursor movement + action
        )
        history = await agent.run(max_steps=10)
        print("Final Result:", history.final_result())
        await context.close()
        await browser.close()

asyncio.run(test_custom_context())