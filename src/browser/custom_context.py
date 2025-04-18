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
        self._emunium = None
        self._emunium_lock = asyncio.Lock()

    async def _ensure_emunium_initialized(self):
        """Lazily initialize EmuniumPlaywright with the current page if not already done."""
        async with self._emunium_lock:
            if self._emunium is None:
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
                
                dom_state = await page.content()
                print(f"DOM state before action: {dom_state[:500]}...")  # Truncate for brevity
                
                # Ensure viewport is set
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
        """Click an element with human-like behavior using emunium."""
        try:
            await self._ensure_emunium_initialized()
            page = await self.get_current_page()
            if page is None:
                raise RuntimeError("No current page available")
            
            # Wait for page stability
            """ await page.wait_for_load_state('networkidle') """
            await page.evaluate('document.body.style.zoom = 1')
            
            element = await page.wait_for_selector(selector, timeout=timeout)
            if not element:
                raise ValueError(f"Element with selector {selector} not found")
            
            # Ensure element is visible and in viewport
            if not await element.is_visible():
                raise ValueError(f"Element {selector} is not visible")
            await element.scroll_into_view_if_needed()
            
            # Get bounding box for debugging
            bounding_box = await element.bounding_box()
            if not bounding_box:
                raise ValueError(f"Element {selector} has no bounding box")
            print(f"Clicking element {selector} with bounding box: {bounding_box}")
            
            await self._emunium.click_at(element)
            logger.info(f"Clicked element {selector}")
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {str(e)}")
            raise

    async def type_at_element(self, selector: str, text: str, timeout: int = 30000):
        """Type text into an element with human-like behavior using emunium."""
        try:
            await self._ensure_emunium_initialized()
            page = await self.get_current_page()
            if page is None:
                raise RuntimeError("No current page available")
            
            # Wait for page stability
            """ await page.wait_for_load_state('networkidle') """
            await page.evaluate('document.body.style.zoom = 1')
            
            element = await page.wait_for_selector(selector, timeout=timeout)
            if not element:
                raise ValueError(f"Element with selector {selector} not found")
            
            # Ensure element is visible and in viewport
            if not await element.is_visible():
                raise ValueError(f"Element {selector} is not visible")
            await element.scroll_into_view_if_needed()
            
            # Get bounding box for dynamic positioning
            bounding_box = await element.bounding_box()
            if not bounding_box:
                raise ValueError(f"Element {selector} has no bounding box")
            print(f"Typing into element {selector} with bounding box: {bounding_box}")
            
            # Calculate center of the element
            x_offset = bounding_box['x'] + bounding_box['width'] / 2
            y_offset = bounding_box['y'] + bounding_box['height'] / 2
            
            # Move to the element
            await self._emunium.move_to(element)
            await self._emunium.type_at(element, text)
            logger.info(f"Typed '{text}' at element {selector} at ({x_offset}, {y_offset})")
        except Exception as e:
            logger.error(f"Error typing at element {selector}: {str(e)}")
            raise
        
    async def scroll_down(self, amount: Optional[int] = None):
        """Scroll down the page by a specified amount or one page with human-like behavior using Emunium."""
        try:
            await self._ensure_emunium_initialized()
            page = await self.get_current_page()
            if page is None:
                raise RuntimeError("No current page available")
            
            # Wait for page stability
            """ await page.wait_for_load_state('networkidle') """
            await page.evaluate('document.body.style.zoom = 1')
            
            # Get window height for one-page scroll
            window_height = await page.evaluate('window.innerHeight')
            scroll_amount = amount if amount is not None else window_height
            
            async with self._emunium_lock:
                # Get current scroll position
                current_scroll_y = await page.evaluate('window.scrollY')
                # Calculate target scroll position
                target_scroll_y = current_scroll_y + scroll_amount
                
                # Use Emunium's scroll_to for human-like scrolling
                await self._emunium.scroll_to(0, target_scroll_y)
                scroll_type = 'Emunium'
            
            amount_str = f'{scroll_amount} pixels' if amount is not None else 'one page'
            logger.info(f"Scrolled down by {amount_str} with {scroll_type}")
            
            # Log scroll position for debugging
            scroll_position = await page.evaluate('window.scrollY')
            print(f"Scroll position after scroll_down: {scroll_position}")
        
        except Exception as e:
            logger.error(f"Error scrolling down: {str(e)}")
            raise