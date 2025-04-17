import asyncio
import logging
import pyperclip
from typing import Optional, Type, Dict, List
from pydantic import BaseModel
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.controller.service import Controller, Context
from browser_use.controller.views import InputTextAction  # Import InputTextAction for param_model
from browser_use.utils import time_execution_sync
from langchain_core.language_models.chat_models import BaseChatModel
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
        async def paste_from_clipboard(browser: CustomBrowserContext):
            text = pyperclip.paste()
            await browser.type_at_element("*:focus", text)  # Type into focused element
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
                await browser.move_to_element(selector)
                await browser.type_at_element(selector, text)
                return ActionResult(extracted_content=f"Typed '{text}' at element {selector}")
            except Exception as e:
                logger.error(f"Failed to type at element: {str(e)}")
                return ActionResult(error=str(e))

        @self.registry.action(
            "Input text into a input interactive element",
            param_model=InputTextAction
        )
        async def input_text(params: InputTextAction, browser, has_sensitive_data: bool = False):
            """Custom input text action overriding browser_use's default."""
            print("Custom Input")  # Verify custom action is used
            try:
                if params.index not in await browser.get_selector_map():
                    raise Exception(f"Element index {params.index} does not exist - retry or use alternative actions")

                element_node = await browser.get_dom_element_by_index(params.index)
                css_selector = browser._enhanced_css_selector_for_element(
                    element_node, include_dynamic_attributes=True
                )
                await browser.type_at_element(css_selector, params.text)
                if not has_sensitive_data:
                    msg = f"⌨️ Custom Input '{params.text}' into index {params.index}"
                else:
                    msg = f"⌨️ Custom Input sensitive data into index {params.index}"
                logger.info(msg)
                logger.debug(f"Element xpath: {element_node.xpath}")
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to input text at index {params.index}: {str(e)}")
                return ActionResult(error=str(e))

    @time_execution_sync('--act_custom')
    async def act_custom(
        self,
        action: ActionModel,
        browser_context: CustomBrowserContext,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[List[str]] = None,
        context: Optional[Context] = None,
    ) -> ActionResult:
        """Execute a custom action using the registry."""
        try:
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
        browser_context: CustomBrowserContext,
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