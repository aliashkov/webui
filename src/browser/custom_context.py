import logging
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from emunium import EmuniumPlaywright  # Import emunium

logger = logging.getLogger(__name__)

class CustomBrowserContext(BrowserContext):
    def __init__(
        self,
        browser: "Browser",
        config: BrowserContextConfig = BrowserContextConfig()
    ):
        super().__init__(browser=browser, config=config)
        self.emunium = None  # Initialize as None; will be set lazily

    async def _ensure_emunium_initialized(self):
        """Lazily initialize EmuniumPlaywright with the current page if not already done."""
        if self.emunium is None:
            page = await self.get_current_page()
            self.emunium = EmuniumPlaywright(page)
            logger.debug("EmuniumPlaywright initialized")

    async def move_to_element(self, selector: str, timeout: int = 30000):
        """Move the mouse to the center of an element with human-like trajectory using emunium."""
        try:
            await self._ensure_emunium_initialized()  # Ensure emunium is ready
            page = await self.get_current_page()
            element = await page.wait_for_selector(selector, timeout=timeout)
            if not element:
                raise ValueError(f"Element with selector {selector} not found")

            # Use emunium to move to the element
            await self.emunium.move_to(element)
        except Exception as e:
            logger.error(f"Error moving to element {selector}: {str(e)}")
            raise

    async def click_element(self, selector: str, timeout: int = 30000):
        """Move to and click an element with human-like behavior using emunium."""
        try:
            await self._ensure_emunium_initialized()  # Ensure emunium is ready
            page = await self.get_current_page()
            element = await page.wait_for_selector(selector, timeout=timeout)
            if not element:
                raise ValueError(f"Element with selector {selector} not found")

            # Use emunium to click the element
            await self.emunium.click_at(element)
            logger.info(f"Clicked element {selector}")
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {str(e)}")
            raise

    async def type_at_element(self, selector: str, text: str, timeout: int = 30000):
        """Move to an element and type text with human-like behavior using emunium."""
        try:
            await self._ensure_emunium_initialized()  # Ensure emunium is ready
            page = await self.get_current_page()
            element = await page.wait_for_selector(selector, timeout=timeout)
            if not element:
                raise ValueError(f"Element with selector {selector} not found")

            # Use emunium to type at the element
            await self.emunium.type_at(element, text)
            logger.info(f"Typed '{text}' at element {selector}")
        except Exception as e:
            logger.error(f"Error typing at element {selector}: {str(e)}")
            raise
