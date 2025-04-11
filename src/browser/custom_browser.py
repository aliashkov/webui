import asyncio
import logging
from playwright.async_api import Browser as PlaywrightBrowser
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig
from .custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)

class CustomBrowser(Browser):
    async def new_context(
        self,
        config: BrowserContextConfig = BrowserContextConfig()
    ) -> CustomBrowserContext:
        return CustomBrowserContext(config=config, browser=self)
