import asyncio
import logging
from typing import Optional, Dict
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from emunium import EmuniumPlaywright
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DOMElementNode:
    tag_name: str
    attributes: Dict[str, str]
    xpath: str
    highlight_index: Optional[int] = None

class CustomBrowserContext(BrowserContext):
    def __init__(
        self,
        browser: "Browser",
        config: BrowserContextConfig = BrowserContextConfig()
    ):
        super().__init__(browser=browser, config=config)
        self._emunium = None  # Use protected attribute
        self._emunium_lock = asyncio.Lock()  # Add lock for thread safety

    async def _ensure_emunium_initialized(self):
        """Lazily initialize EmuniumPlaywright with the current page if not already done."""
        """ if self._emunium is None:
            async with self._emunium_lock:
                if self._emunium is None: """
        page = await self.get_current_page()
        if page is None:
            raise RuntimeError("No current page available")
                    
        # Get browser window dimensions
        window_dimensions = await page.evaluate('''() => {
           return {
                'width': window.outerWidth,
                'height': window.outerHeight
            };
        }''')
                    
                    # Ensure viewport is set before creating Emunium instance
        if page.viewport_size is None:
            await page.set_viewport_size({
                'width': window_dimensions['width'],
                'height': window_dimensions['height']
            })
            print(f"Set viewport size to window dimensions: {window_dimensions}")
        else:
            logger.debug(f"Current viewport size: {page.viewport_size}")
                    
        self._emunium = EmuniumPlaywright(page)
        print("EmuniumPlaywright initialized with viewport size: %s", page.viewport_size)

    async def click_element(self, selector: str, timeout: int = 30000):
        """Move to and click an element with human-like behavior using emunium."""
        try:
            await self._ensure_emunium_initialized()  # Ensure emunium is ready
            page = await self.get_current_page()
            element = await page.wait_for_selector(selector, timeout=timeout)
            if not element:
                raise ValueError(f"Element with selector {selector} not found")

            # Use emunium to click the element
            await self._emunium.click_at(element)  # Fixed: Use _emunium
            logger.info(f"Clicked element {selector}")
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {str(e)}")
            raise

    async def type_at_element(self, selector: str, text: str, timeout: int = 30000):
        """Move to an element and type text with human-like behavior using emunium."""
        try:
            await self._ensure_emunium_initialized()
            page = await self.get_current_page()
            element = await page.wait_for_selector(selector, timeout=timeout)
            if not element:
                raise ValueError(f"Element with selector {selector} not found")

   
            # Then type at the element
            await self._emunium.type_at(element, text)
            logger.info(f"Typed '{text}' at element {selector}")
        except Exception as e:
            logger.error(f"Error typing at element {selector}: {str(e)}")
            raise
