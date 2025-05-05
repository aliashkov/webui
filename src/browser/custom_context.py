import asyncio
import logging
import random
import pyautogui
from typing import Optional, Dict
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from src.emunium.browsers import EmuniumPlaywright  # Import from your custom emunium folder
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
                    logger.error("No current page available for Emunium initialization")
                    raise RuntimeError("No current page available")
                
                # Get browser window dimensions
                window_dimensions = await page.evaluate('''() => {
                    return {
                        'width': window.innerWidth,
                        'height': window.innerHeight
                    };
                }''')
                logger.debug(f"Browser window dimensions: {window_dimensions}")
                
                dom_state = await page.content()
                print(f"DOM state before action: {dom_state[:500]}...")  # Truncate for brevity
                
                # Ensure viewport is set
                if page.viewport_size is None:
                    await page.set_viewport_size({
                        'width': window_dimensions['width'],
                        'height': window_dimensions['height']
                    })
                    logger.info(f"Set viewport size to window dimensions: {window_dimensions}")
                else:
                    logger.debug(f"Current viewport size: {page.viewport_size}")
                
                try:
                    self._emunium = EmuniumPlaywright(page)
                    logger.info("EmuniumPlaywright initialized with viewport size: %s", page.viewport_size)
                except Exception as e:
                    logger.error(f"Failed to initialize EmuniumPlaywright: {str(e)}")
                    self._emunium = None
                    raise
                
            else:
                logger.debug("Emunium already initialized")

    async def _scroll_by_pixels(self, scroll_amount: int):
        """Scroll the page by a specified pixel amount with human-like behavior."""
        try:
            page = await self.get_current_page()
            if page is None:
                raise RuntimeError("No current page available")
            
            # Ensure browser window is focused
            await page.bring_to_front()
            logger.debug("Brought browser window to front")
            
            # Check for custom scroll container
            scroll_container = await page.evaluate('''() => {
                const container = document.querySelector('div[style*="overflow: auto"], div[style*="overflow-y: scroll"], main[style*="overflow: auto"], section[style*="overflow: auto"]') || document.body;
                return {
                    tag: container.tagName,
                    id: container.id,
                    class: container.className,
                    scrollHeight: container.scrollHeight,
                    clientHeight: container.clientHeight
                };
            }''')
            logger.debug(f"Scroll container: {scroll_container}")
            
            async with self._emunium_lock:
                scroll_type = 'Playwright'
                if self._emunium:
                    logger.debug("Using Emunium for scrolling")
                    # Simulate human-like scrolling with small steps
                    step_size = 100  # Scroll 100 pixels per step
                    scroll_steps = abs(scroll_amount) // step_size
                    scroll_direction = -1 if scroll_amount > 0 else 1  # Negative for down, positive for up
                    
                    for _ in range(scroll_steps):
                        pyautogui.scroll(scroll_direction * step_size)
                        await asyncio.sleep(random.uniform(0.05, 0.1))  # Random delay for realism
                    
                    remaining_scroll = abs(scroll_amount) % step_size
                    if remaining_scroll:
                        pyautogui.scroll(scroll_direction * remaining_scroll)
                        await asyncio.sleep(random.uniform(0.05, 0.1))
                    
                    scroll_type = 'Emunium'
                else:
                    logger.debug("Falling back to Playwright scrolling")
                    # Scroll the appropriate container
                    await page.evaluate(f'window.scrollBy(0, {scroll_amount});')
                
                logger.debug(f"Scroll type: {scroll_type}")
                return scroll_type
        
        except Exception as e:
            logger.error(f"Error in _scroll_by_pixels: {str(e)}")
            raise

    async def scroll_down(self, amount: Optional[int] = None):
        """Scroll down the page by a specified amount or one page with human-like behavior."""
        try:
            await self._ensure_emunium_initialized()
            page = await self.get_current_page()
            if page is None:
                raise RuntimeError("No current page available")
            
            await page.evaluate('document.body.style.zoom = 1')
            
            # Get window height for one-page scroll
            window_height = await page.evaluate('window.innerHeight')
            scroll_amount = amount if amount is not None else window_height
            
            # Perform the scroll
            scroll_type = await self._scroll_by_pixels(scroll_amount)
            
            amount_str = f'{scroll_amount} pixels' if amount is not None else 'one page'
            logger.info(f"Scrolled down by {amount_str} with {scroll_type}")
            
            # Log scroll position for debugging
            scroll_position = await page.evaluate('window.scrollY')
            scroll_container_position = await page.evaluate('''() => {
                const container = document.querySelector('div[style*="overflow: auto"], div[style*="overflow-y: scroll"], main[style*="overflow: auto"], section[style*="overflow: auto"]') || document.body;
                return container.scrollTop;
            }''')
            print(f"Scroll position after scroll_down: window.scrollY={scroll_position}, container.scrollTop={scroll_container_position}")
            logger.debug(f"Scroll position: window.scrollY={scroll_position}, container.scrollTop={scroll_container_position}")
            
            # Verify scrollability
            scroll_height = await page.evaluate('document.body.scrollHeight')
            logger.debug(f"Page scrollHeight: {scroll_height}, window.innerHeight: {window_height}")
            if scroll_position == 0 and scroll_container_position == 0 and scroll_amount > 0:
                logger.warning("Scroll position did not change; page may not be scrollable or container issue")
        
        except Exception as e:
            logger.error(f"Error scrolling down: {str(e)}")
            raise

    async def scroll_up(self, amount: Optional[int] = None):
        """Scroll up the page by a specified amount or one page with human-like behavior."""
        try:
            await self._ensure_emunium_initialized()
            page = await self.get_current_page()
            if page is None:
                raise RuntimeError("No current page available")
            
            # Wait for page stability
            await page.evaluate('document.body.style.zoom = 1')
            
            # Get window height for one-page scroll
            window_height = await page.evaluate('window.innerHeight')
            scroll_amount = -(amount if amount is not None else window_height)  # Negative for scrolling up
            
            # Perform the scroll
            scroll_type = await self._scroll_by_pixels(scroll_amount)
            
            amount_str = f'{abs(scroll_amount)} pixels' if amount is not None else 'one page'
            logger.info(f"Scrolled up by {amount_str} with {scroll_type}")
            
            # Log scroll position for debugging
            scroll_position = await page.evaluate('window.scrollY')
            scroll_container_position = await page.evaluate('''() => {
                const container = document.querySelector('div[style*="overflow: auto"], div[style*="overflow-y: scroll"], main[style*="overflow: auto"], section[style*="overflow: auto"]') || document.body;
                return container.scrollTop;
            }''')
            print(f"Scroll position after scroll_up: window.scrollY={scroll_position}, container.scrollTop={scroll_container_position}")
            logger.debug(f"Scroll position: window.scrollY={scroll_position}, container.scrollTop={scroll_container_position}")
            
            # Verify scrollability
            scroll_height = await page.evaluate('document.body.scrollHeight')
            logger.debug(f"Page scrollHeight: {scroll_height}, window.innerHeight: {window_height}")
            if scroll_position == scroll_container_position == 0 and scroll_amount < 0:
                logger.warning("Scroll position is at top; page may already be at the top or container issue")
        
        except Exception as e:
            logger.error(f"Error scrolling up: {str(e)}")
            raise

    async def click_element(self, selector: str, timeout: int = 30000, retries: int = 3):
        """Click an element with human-like behavior using emunium, with fallback to Playwright."""
        for attempt in range(retries):
            try:
                await self._ensure_emunium_initialized()
                page = await self.get_current_page()
                if page is None:
                    raise RuntimeError("No current page available")

                # Wait for page stability
                await page.evaluate('document.body.style.zoom = 1')

                # Wait for the element to be visible
                element = await page.wait_for_selector(selector, state="visible", timeout=timeout)
                if not element:
                    raise ValueError(f"Element with selector {selector} not found")

                # Ensure element is visible and in viewport
                if not await element.is_visible():
                    raise ValueError(f"Element {selector} is not visible")
                await element.scroll_into_view_if_needed()

                # Get bounding box and convert coordinates to integers
                """ bounding_box = await element.bounding_box()
                if not bounding_box:
                    raise ValueError(f"Element {selector} has no bounding box")

                # Round coordinates to integers
                bounding_box = {
                    'x': int(bounding_box['x']),
                    'y': int(bounding_box['y']),
                    'width': int(bounding_box['width']),
                    'height': int(bounding_box['height'])
                }
                logger.debug(f"Clicking element {selector} with bounding box: {bounding_box}")

                # Check if element is within viewport
                viewport_height = page.viewport_size['height']
                if bounding_box['y'] + bounding_box['height'] > viewport_height:
                    logger.warning(f"Element at y={bounding_box['y']} exceeds viewport height={viewport_height}")
                    await page.evaluate(f"window.scrollBy(0, {bounding_box['y'] + bounding_box['height'] - viewport_height + 10});")
                    bounding_box = await element.bounding_box()  # Recompute after scrolling
                    logger.debug(f"Adjusted bounding box after scroll: {bounding_box}") """

                # Try Emunium click
                try:
                    await self._emunium.scroll_to(element)
                    await self._emunium.click_at(element)
                    logger.info(f"Clicked element {selector} with Emunium")
                    return
                except Exception as emunium_error:
                    logger.warning(f"Emunium click failed: {str(emunium_error)}. Falling back to Playwright click.")

                    # Fallback to Playwright click
                    await element.click()
                    logger.info(f"Clicked element {selector} with Playwright")
                    return

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for selector {selector}: {str(e)}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(random.uniform(0.5, 1.0))  # Random delay before retry

        raise RuntimeError(f"Failed to click element {selector} after {retries} attempts")

    async def type_at_element(self, selector: str, text: str, timeout: int = 30000, enableEnter: bool = True):
        """Type text into an element with human-like behavior using emunium, ensuring the field is cleared first."""
        try:
            await self._ensure_emunium_initialized()
            page = await self.get_current_page()
            if page is None:
                raise RuntimeError("No current page available")

            # Wait for page stability (optional but sometimes helpful)
            # await page.wait_for_load_state('networkidle') # Consider using this if needed
            await page.evaluate('document.body.style.zoom = 1')

            element = await page.wait_for_selector(selector, timeout=timeout) # Wait for visible
            if not element:
                raise ValueError(f"Element with selector {selector} not found")

            # Ensure element is visible and in viewport
            # The state="visible" above helps, but scroll_into_view is still good practice
            await element.scroll_into_view_if_needed()
            await asyncio.sleep(0.2)

            logger.debug(f"Clearing input field: {selector}")
            """ await element.fill("") """

            # Get bounding box (for logging/debugging, not strictly needed for the fix)
            bounding_box = await element.bounding_box()
            if not bounding_box:
                # This shouldn't happen if wait_for_selector succeeded, but check just in case
                 logger.warning(f"Element {selector} has no bounding box after clearing/scrolling.")
                 # Attempt focus before proceeding
                 await element.focus()
                 bounding_box = await element.bounding_box()
                 if not bounding_box:
                    raise ValueError(f"Element {selector} still has no bounding box")
            print(f"Typing into element {selector} with bounding box: {bounding_box}")


            await element.focus()
            await asyncio.sleep(0.1)

            # Clear the content using JavaScript
            await element.evaluate('el => { el.textContent = ""; }')
            
            
            await self._emunium.move_to(element) # Move mouse over the element
            await self._emunium.type_at(element, text) # Type the text

            if enableEnter:
               await self._emunium.type_at(element, '\n')

            logger.info(f"Typed '{text}' at element {selector}")
            
        except Exception as e:
            # Improve error logging
            current_url = await page.url() if page else "Unknown URL"
            logger.error(f"Error typing at element '{selector}' on page '{current_url}': {str(e)}", exc_info=True)
            # Optionally capture screenshot on error
            # try:
            #     if page: await page.screenshot(path="error_screenshot_typing.png")
            # except Exception as screen_err:
            #     logger.error(f"Failed to take screenshot on error: {screen_err}")
            raise