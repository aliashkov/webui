import logging
import pyperclip
from typing import Optional, Type
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.controller.service import Controller
from browser_use.browser.context import BrowserContext
from src.browser.custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)

class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [], output_model: Optional[Type[BaseModel]] = None):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()

    def _register_custom_actions(self):
        """Register all custom browser actions."""
        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            page = await browser.get_current_page()
            await page.keyboard.type(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Move cursor to element")
        async def move_cursor_to_element(browser: CustomBrowserContext, selector: str):
            """Move the cursor to an element with human-like trajectory."""
            try:
                await browser.move_to_element(selector)
                return ActionResult(extracted_content=f"Moved cursor to element with selector {selector}")
            except Exception as e:
                logger.error(f"Failed to move cursor to element: {str(e)}")
                return ActionResult(error=str(e))

        @self.registry.action("Click element with human behavior")
        async def click_element_human(browser: CustomBrowserContext, selector: str):
            """Move cursor to and click an element with human-like behavior."""
            try:
                # Move cursor first
                await browser.move_to_element(selector)
                await browser.click_element(selector)
                return ActionResult(extracted_content=f"Clicked element with selector {selector}")
            except Exception as e:
                logger.error(f"Failed to click element: {str(e)}")
                return ActionResult(error=str(e))

        @self.registry.action("Type text at element")
        async def type_text_at_element(browser: CustomBrowserContext, selector: str, text: str):
            """Move cursor to an element and type text with human-like behavior."""
            try:
                # Move cursor first
                await browser.move_to_element(selector)
                await browser.type_at_element(selector, text)
                return ActionResult(extracted_content=f"Typed '{text}' at element {selector}")
            except Exception as e:
                logger.error(f"Failed to type at element: {str(e)}")
                return ActionResult(error=str(e))