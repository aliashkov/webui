import asyncio
import logging
import pyperclip
from typing import Optional, Type, Dict, List
from pydantic import BaseModel
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.controller.service import Controller, Context
from browser_use.controller.views import InputTextAction, ClickElementAction  # Import InputTextAction for param_model
from browser_use.utils import time_execution_sync
from langchain_core.language_models.chat_models import BaseChatModel
from src.browser.custom_context import CustomBrowserContext
from browser_use.browser.context import BrowserContext
from emunium import EmuniumPlaywright


logger = logging.getLogger(__name__)

class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [], output_model: Optional[Type[BaseModel]] = None):
        self._emunium = None  # Use protected attribute
        self._emunium_lock = asyncio.Lock()  # Add lock for thread safety
        self.browserContextOpt = None
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
    def _register_custom_actions(self):
        """Register all custom browser actions."""
        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: CustomBrowserContext):
            print("Custom Input 55555555555")
            text = pyperclip.paste()
            await browser.type_at_element("*:focus", text)  # Type into focused element
            return ActionResult(extracted_content=text)

        @self.registry.action("Move cursor to element")
        async def move_cursor_to_element(browser: CustomBrowserContext, selector: str):
            """Move the cursor to an element with human-like trajectory."""
            try:
                """ await browser.move_to_element(selector) """
                return ActionResult(extracted_content=f"Moved cursor to element with selector {selector}")
            except Exception as e:
                logger.error(f"Failed to move cursor to element: {str(e)}")
                return ActionResult(error=str(e))

        @self.registry.action("Click element with human behavior")
        async def click_element_human(browser: CustomBrowserContext, selector: str):
            """Move cursor to and click an element with human-like behavior."""
            try:
                """ await browser.move_to_element(selector) """
                await browser.click_element(selector)
                return ActionResult(extracted_content=f"Clicked element with selector {selector}")
            except Exception as e:
                logger.error(f"Failed to click element: {str(e)}")
                return ActionResult(error=str(e))

        @self.registry.action("Type text at element")
        async def type_text_at_element(browser: CustomBrowserContext, selector: str, text: str):
            """Move cursor to an element and type text with human-like behavior."""
            try:
                """ await browser.move_to_element(selector)
                await browser.type_at_element(selector, text) """
                return ActionResult(extracted_content=f"Typed '{text}' at element {selector}")
            except Exception as e:
                logger.error(f"Failed to type at element: {str(e)}")
                return ActionResult(error=str(e))
            
            
            
        @self.registry.action("Click element", param_model=ClickElementAction)
        async def click_element(params: ClickElementAction, browser: CustomBrowserContext):
            """Click an element by index, with emunium support for alternative clicking methods."""
            try:
                async with self._emunium_lock:  # Ensure thread safety for emunium access
                    session = await browser.get_session()

                    if params.index not in await browser.get_selector_map():
                        raise Exception(f"Element with index {params.index} does not exist - retry or use alternative actions")

                    element_node = await browser.get_dom_element_by_index(params.index)
                    initial_pages = len(session.context.pages)

                    # Check if element is a file uploader
                    if await browser.is_file_uploader(element_node):
                        msg = f"Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files"
                        logger.info(msg)
                        return ActionResult(extracted_content=msg, include_in_memory=True)

                    msg = None

                    if self._emunium:
                        # Use emunium-specific clicking method with enhanced CSS selector
                        css_selector = browser._enhanced_css_selector_for_element(
                            element_node, include_dynamic_attributes=True
                        )
                        await browser.click_element(css_selector)
                        msg = f"🖱️ Emunium Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}"
                    else:
                        # Default clicking method
                        download_path = await browser._click_element_node(element_node)
                        if download_path:
                            msg = f"💾 Downloaded file to {download_path}"
                        else:
                            msg = f"🖱️ Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}"

                    logger.info(msg)
                    logger.debug(f"Element xpath: {element_node.xpath}")

                    # Handle new tab if opened
                    if len(session.context.pages) > initial_pages:
                        new_tab_msg = "New tab opened - switching to it"
                        msg += f" - {new_tab_msg}"
                        logger.info(new_tab_msg)
                        await browser.switch_to_tab(-1)

                    return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.warning(f"Element not clickable with index {params.index} - most likely the page changed: {str(e)}")
                return ActionResult(error=str(e))    

        @self.registry.action(
            "Input text into a input interactive element",
            param_model=InputTextAction
        )
        
        
        
        async def input_text(params: InputTextAction, browser, has_sensitive_data: bool = False):
            """Custom input text action overriding browser_use's default."""
            print("Custom Input")  # Verify custom action is used
            print("Browser Context 5555", self.browserContextOpt)
            try:
                if params.index not in await browser.get_selector_map():
                    raise Exception(f"Element index {params.index} does not exist - retry or use alternative actions")

                element_node = await browser.get_dom_element_by_index(params.index)
                print("Self emunium", self._emunium_lock)
                print("Emunium", self._emunium)
                
                if self._emunium:
                    css_selector = browser._enhanced_css_selector_for_element(
                        element_node, include_dynamic_attributes=True
                    )
                    await browser.type_at_element(css_selector, params.text) # type: ignore
                    if not has_sensitive_data:
                        msg = f"⌨️ Custom Input '{params.text}' into index {params.index}"
                    else:
                        msg = f"⌨️ Custom Input sensitive data into index {params.index}"
                    logger.info(msg)
                    logger.debug(f"Element xpath: {element_node.xpath}")
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    await browser._input_text_element_node(element_node, params.text)
                    if not has_sensitive_data:
                        msg = f'⌨️  Input {params.text} into index {params.index}'
                    else:
                        msg = f'⌨️  Input sensitive data into index {params.index}'
                        logger.info(msg)
                        logger.debug(f'Element xpath: {element_node.xpath}')
                        return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to input text at index {params.index}: {str(e)}")
                return ActionResult(error=str(e))

    @time_execution_sync('--act_custom')
    async def act_custom(
        self,
        action: ActionModel,
        browser_context: BrowserContext,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[List[str]] = None,
        context: Optional[Context] = None,
        enable_emunium=False,  # Add comma here
        browserContextOpt: Optional[CustomBrowserContext] = None,
    ) -> ActionResult:
        """Execute a custom action using the registry."""
        try:
            print("Custom browser context", browserContextOpt)
            if browserContextOpt:
                self.browserContextOpt = browserContextOpt
                """ page = browserContext.move_to_element() # type: ignore
                print("Page", page.viewport_size) # type: ignore """
            print("Enable emunium", enable_emunium)
            if enable_emunium:
                self._emunium = enable_emunium
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    result = await self.registry.execute_action(
                        action_name=action_name,
                        params=params,
                        browser=browser_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                        context=context,
                    )

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f"Invalid action result type: {type(result)} of {result}")
            return ActionResult()
        except Exception as e:
            logger.error(f"Action execution failed: {str(e)}")
            return ActionResult(error=str(e))

    async def multi_act_custom(
        self,
        actions: List[ActionModel],
        browser_context: BrowserContext,
        check_for_new_elements: bool = True,
    ) -> List[ActionResult]:
        """Execute multiple actions."""
        results = []
        
        cached_selector_map = await browser_context.get_selector_map()
        cached_path_hashes = set(e.hash.branch_path_hash for e in cached_selector_map.values())

        await browser_context.remove_highlights()

        for i, action in enumerate(actions):
            if action.get_index() is not None and i != 0:
                new_state = await browser_context.get_state()
                new_path_hashes = set(e.hash.branch_path_hash for e in new_state.selector_map.values())
                if check_for_new_elements and not new_path_hashes.issubset(cached_path_hashes):
                    msg = f"Something new appeared after action {i} / {len(actions)}"
                    logger.info(msg)
                    results.append(ActionResult(extracted_content=msg, include_in_memory=True))
                    break

            result = await self.act_custom(
                action=action,
                browser_context=browser_context,
                page_extraction_llm=None,
                sensitive_data=None,
                available_file_paths=None,
                context=None,
            )

            results.append(result)

            logger.debug(f"Executed action {i + 1} / {len(actions)}")
            if result.is_done or result.error or i == len(actions) - 1:
                break

            await asyncio.sleep(browser_context.config.wait_between_actions)

        return results