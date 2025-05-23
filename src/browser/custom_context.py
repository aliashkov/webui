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

    async def click_element(self, selector: str, timeout: int = 5000, retries: int = 2):
        for attempt in range(retries):
            try:
                await self._ensure_emunium_initialized()
                page = await self.get_current_page()
                if page is None:
                    raise RuntimeError("No current page available")

                await page.evaluate('document.body.style.zoom = 1')
                element = await page.wait_for_selector(selector, state="visible", timeout=timeout) # Убедимся, что он видим
                if not element: # wait_for_selector сам вызовет TimeoutError, но для ясности
                    raise ValueError(f"Element with selector {selector} not found or not visible")

                if not await element.is_visible():
                    raise ValueError(f"Element {selector} is not visible")

                # Playwright's scroll, если нужен (может быть избыточен, если Emunium скроллит)
                await element.scroll_into_view_if_needed()
                await asyncio.sleep(0.1) # Небольшая пауза после Playwright скролла

                # Получаем bounding_box ПЕРЕД Emunium скроллом для передачи в Emunium.scroll_to,
                # если Emunium.scroll_to ожидает сам элемент, а не его bounding_box,
                # то он внутри себя должен получить bounding_box.
                # bounding_box_before_emunium_scroll = await element.bounding_box()
                # if not bounding_box_before_emunium_scroll:
                #     raise ValueError(f"Element {selector} has no bounding box before emunium scroll")

                logger.debug(f"Attempting Emunium scroll for {selector}")
                await self._emunium.scroll_to(element) # Emunium скроллит используя pyautogui
                
                # ВАЖНО: Даем браузеру время "осознать" скролл и перерисоваться
                await asyncio.sleep(0.3) # Можно подобрать значение, 0.2-0.5с

                # ВАЖНО: После системного скролла (pyautogui) Playwright может иметь
                # устаревшие данные о положении элемента. Получаем свежий bounding_box.
                current_bounding_box = await element.bounding_box()
                if not current_bounding_box:
                    # Это маловероятно, если элемент был видим и скролл прошел, но для надежности
                    logger.warning(f"Element {selector} has no bounding box AFTER emunium scroll. Retrying Playwright click.")
                    await element.click(timeout=5000) # Быстрый Playwright клик как fallback
                    return
                
                # Округляем координаты
                current_bounding_box = {
                    'x': int(current_bounding_box['x']),
                    'y': int(current_bounding_box['y']),
                    'width': int(current_bounding_box['width']),
                    'height': int(current_bounding_box['height'])
                }
                logger.debug(f"Clicking element {selector} with FRESH bounding box after emunium scroll: {current_bounding_box}")
                
                # Проверка, находится ли элемент в пределах вьюпорта (после Emunium скролла)
                # viewport_size = page.viewport_size # {'width': ..., 'height': ...}
                # if not (0 <= current_bounding_box['y'] < viewport_size['height'] and \
                #         0 <= current_bounding_box['y'] + current_bounding_box['height'] <= viewport_size['height']):
                #     logger.warning(f"Element {selector} y-coords {current_bounding_box['y']}-{current_bounding_box['y']+current_bounding_box['height']} partially or fully outside viewport height {viewport_size['height']} after Emunium scroll. Attempting Playwright click.")
                #     await element.click(timeout=5000) # Быстрый Playwright клик
                #     return

                try:
                    # Передаем сам элемент, т.к. click_at внутри себя должен взять свежий bounding_box
                    # или, если click_at ожидает box, передаем current_bounding_box
                    await self._emunium.click_at(element) # Предполагается, что click_at внутри возьмет свежий bounding_box
                    logger.info(f"Clicked element {selector} with Emunium")
                    return
                except Exception as emunium_error:
                    logger.warning(f"Emunium click failed: {str(emunium_error)}. Falling back to Playwright click.")
                    await element.click() # Playwright клик как fallback
                    logger.info(f"Clicked element {selector} with Playwright fallback")
                    return

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} to click {selector} failed: {str(e)}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(random.uniform(0.5, 1.0))

        raise RuntimeError(f"Failed to click element {selector} after {retries} attempts")

    async def type_at_element(self, selector: str, text: str, timeout: int = 30000, enableEnter: bool = True):
        page = await self.get_current_page()
        if page is None:
            raise RuntimeError("No current page available. Ensure browser and page are initialized.")

        try:
            await self._ensure_emunium_initialized()

            logger.debug(f"Attempting to type '{text}' into element '{selector}'")

            await page.evaluate('document.body.style.zoom = 1')
            logger.debug("Page zoom reset to 1.")
            await asyncio.sleep(0.05)

            logger.debug(f"Waiting for selector '{selector}' to be visible.")
            element = await page.wait_for_selector(selector, state="visible", timeout=timeout)
            logger.debug(f"Element '{selector}' found.")

            await element.scroll_into_view_if_needed()
            await asyncio.sleep(0.2) # Небольшая пауза после скролла
            logger.debug(f"Scrolled '{selector}' into view if needed.")

            logger.debug(f"Focusing element: '{selector}'")
            try:
                await element.focus(timeout=5000) # Добавим таймаут на фокус
            except Exception as focus_err:
                logger.warning(f"Could not focus element '{selector}': {focus_err}. Proceeding, but typing might fail.")
            await asyncio.sleep(0.1)

            # --- Модифицированный блок очистки и проверки ---
            tag_name = await element.evaluate("el => el.tagName.toUpperCase()")
            is_content_editable = await element.evaluate("el => el.isContentEditable")

            if tag_name in ["INPUT", "TEXTAREA"] or is_content_editable:
                logger.debug(f"Clearing field '{selector}' using element.fill('')")
                await element.fill("")
                await asyncio.sleep(0.1)

                current_value_for_check = ""
                if tag_name in ["INPUT", "TEXTAREA"]:
                    current_value_for_check = await element.input_value()
                elif is_content_editable:
                    current_value_for_check = await element.text_content() or ""

                if current_value_for_check.strip() != "":
                    logger.warning(f"Field '{selector}' not empty after fill(''). Current value/text: '{current_value_for_check}'. Attempting JS clear.")
                    if tag_name in ["INPUT", "TEXTAREA"]:
                        await element.evaluate('el => { el.value = ""; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); }')
                    elif is_content_editable:
                        await element.evaluate('el => { el.innerHTML = ""; el.dispatchEvent(new Event("input", { bubbles: true })); }')
                    await asyncio.sleep(0.1)

                    if tag_name in ["INPUT", "TEXTAREA"]:
                        current_value_for_check = await element.input_value()
                    elif is_content_editable:
                        current_value_for_check = await element.text_content() or ""

                    if current_value_for_check.strip() != "":
                        logger.error(f"Field '{selector}' could not be cleared. Value/text remains: '{current_value_for_check}'")
                    else:
                        logger.debug(f"Field '{selector}' cleared successfully with JS fallback.")
                else:
                    logger.debug(f"Field '{selector}' cleared successfully with fill('').")
            else:
                logger.info(f"Element '{selector}' (tag: {tag_name}) is not a standard input, textarea, or contenteditable. Skipping standard clear. Emunium will attempt to type.")
            # --- Конец модифицированного блока ---

            # !!! ВАЖНОЕ ИЗМЕНЕНИЕ НИЖЕ !!!
            bounding_box = await element.bounding_box()
            if not bounding_box:
                logger.error(f"Element '{selector}' has NO BOUNDING BOX after operations. Emunium cannot interact with it. This often means the element is hidden, not truly interactive (e.g., a placeholder requiring a click first), or the UI changed.")
                raise RuntimeError(f"Element '{selector}' is not interactable (no bounding box). Ensure it's the correct target and visible.")
            else:
                logger.debug(f"Element '{selector}' bounding box: {bounding_box}")

            logger.debug(f"Moving mouse via emunium to: '{selector}'")
            await self._emunium.move_to(element)

            logger.debug(f"Typing text via emunium into: '{selector}'")
            await self._emunium.type_at(element, text)

            if enableEnter:
               logger.debug(f"Pressing Enter via emunium at: '{selector}'")
               await self._emunium.type_at(element, '\n') # Emunium должен понимать '\n' как Enter

            logger.info(f"Successfully typed '{text}' at element '{selector}'" + (" and pressed Enter." if enableEnter else "."))

        except Exception as e:
            current_url = "Unknown URL (page object not available or failed before URL retrieval)"
            if page:
                try:
                    current_url = page.url
                except Exception as page_url_err:
                    logger.warning(f"Could not get current URL during error handling: {page_url_err}")
            logger.error(f"Error in type_at_element for selector '{selector}' on page '{current_url}': {str(e)}", exc_info=True)
            raise