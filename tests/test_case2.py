import asyncio
import sys
import os
from browser_use import BrowserConfig
from playwright.async_api import async_playwright

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.browser.custom_browser import CustomBrowser
from browser_use.browser.context import BrowserContextConfig

async def test_custom_context():
    async with async_playwright() as p:
        browser = CustomBrowser(config=BrowserConfig(
                    headless=False
                ))
        context = await browser.new_context(config=BrowserContextConfig())
        page = await context.get_current_page()
        await page.goto("https://duckduckgo.com/")
        await context.move_to_element('[data-state="suggesting2"]')
        await context.type_at_element('[data-state="suggesting2"]', "Testing emunium")
        """ await context.click_element('[aria-label="Search"]') """
        await asyncio.sleep(2)  # Observe result
        await context.close()
        await browser.close()

asyncio.run(test_custom_context())