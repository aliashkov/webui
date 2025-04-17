import json
import logging
import pdb
import traceback
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Type, TypeVar
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io
import asyncio
import time
import platform
import re
from browser_use import ActionModel
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.service import Agent
from browser_use.agent.message_manager.utils import convert_input_messages, extract_json_from_model_output, \
    save_conversation
from browser_use.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentState,
    AgentStepInfo,
    StepMetadata,
    ToolCallingMethod,
)
from browser_use.agent.gif import create_history_gif
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserStateHistory
from browser_use.controller.service import Controller
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.agent.prompts import PlannerPrompt

from json_repair import repair_json
from src.utils.agent_state import AgentState

from .custom_message_manager import CustomMessageManager, CustomMessageManagerSettings
from .custom_views import CustomAgentOutput, CustomAgentStepInfo, CustomAgentState

from src.controller.custom_controller import CustomController

from src.browser.custom_context import CustomBrowserContext  # Import CustomBrowserContext


logger = logging.getLogger(__name__)

Context = TypeVar('Context')

from emunium import EmuniumPlaywright


class CustomAgent(Agent):
    def __init__(
            self,
            task: str,
            llm: BaseChatModel,
            add_infos: str = "",
            # Optional parameters
            browser: Browser | None = None,
            browser_context: BrowserContext | None = None,
            controller: Controller[Context] = Controller(),
            # Initial agent run parameters
            sensitive_data: Optional[Dict[str, str]] = None,
            initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
            # Cloud Callbacks
            register_new_step_callback: Callable[['BrowserState', 'AgentOutput', int], Awaitable[None]] | None = None,
            register_done_callback: Callable[['AgentHistoryList'], Awaitable[None]] | None = None,
            register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
            # Agent settings
            use_vision: bool = True,
            use_vision_for_planner: bool = False,
            save_conversation_path: Optional[str] = None,
            save_conversation_path_encoding: Optional[str] = 'utf-8',
            max_failures: int = 3,
            retry_delay: int = 10,
            system_prompt_class: Type[SystemPrompt] = SystemPrompt,
            agent_prompt_class: Type[AgentMessagePrompt] = AgentMessagePrompt,
            max_input_tokens: int = 128000,
            validate_output: bool = False,
            message_context: Optional[str] = None,
            generate_gif: bool | str = False,
            available_file_paths: Optional[list[str]] = None,
            include_attributes: list[str] = [
                'title',
                'type',
                'name',
                'role',
                'aria-label',
                'placeholder',
                'value',
                'alt',
                'aria-expanded',
                'data-date-format',
            ],
            max_actions_per_step: int = 10,
            tool_calling_method: Optional[ToolCallingMethod] = 'auto',
            page_extraction_llm: Optional[BaseChatModel] = None,
            planner_llm: Optional[BaseChatModel] = None,
            planner_interval: int = 1,  # Run planner every N steps
            # Inject state
            injected_agent_state: Optional[AgentState] = None,
            context: Context | None = None,
    ):
        if controller is None:
            controller = CustomController()
        super(CustomAgent, self).__init__(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            sensitive_data=sensitive_data,
            initial_actions=initial_actions,
            register_new_step_callback=register_new_step_callback,
            register_done_callback=register_done_callback,
            register_external_agent_status_raise_error_callback=register_external_agent_status_raise_error_callback,
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            page_extraction_llm=page_extraction_llm,
            planner_llm=planner_llm,
            planner_interval=planner_interval,
            injected_agent_state=injected_agent_state,
            context=context,
        )
        self.state = injected_agent_state or CustomAgentState()
        self.add_infos = add_infos
        self._message_manager = CustomMessageManager(
            task=task,
            system_message=self.settings.system_prompt_class(
                self.available_actions,
                max_actions_per_step=self.settings.max_actions_per_step,
            ).get_system_message(),
            settings=CustomMessageManagerSettings(
                max_input_tokens=self.settings.max_input_tokens,
                include_attributes=self.settings.include_attributes,
                message_context=self.settings.message_context,
                sensitive_data=sensitive_data,
                available_file_paths=self.settings.available_file_paths,
                agent_prompt_class=agent_prompt_class
            ),
            state=self.state.message_manager_state,
        )
        self.last_cursor_selector = None

    def _log_response(self, response: CustomAgentOutput) -> None:
        """Log the model's response"""
        if "Success" in response.current_state.evaluation_previous_goal:
            emoji = "✅"
        elif "Failed" in response.current_state.evaluation_previous_goal:
            emoji = "❌"
        else:
            emoji = "🤷"

        logger.info(f"{emoji} Eval: {response.current_state.evaluation_previous_goal}")
        logger.info(f"🧠 New Memory: {response.current_state.important_contents}")
        logger.info(f"🤔 Thought: {response.current_state.thought}")
        logger.info(f"🎯 Next Goal: {response.current_state.next_goal}")
        for i, action in enumerate(response.action):
            logger.info(
                f"🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}"
            )

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = CustomAgentOutput.type_with_custom_actions(self.ActionModel)
        
        
        
     	# @observe(name='controller.multi_act')
	
    def update_step_info(
            self, model_output: CustomAgentOutput, step_info: CustomAgentStepInfo = None
    ):
        """
        update step info
        """
        if step_info is None:
            return

        step_info.step_number += 1
        important_contents = model_output.current_state.important_contents
        if (
                important_contents
                and "None" not in important_contents
                and important_contents not in step_info.memory
        ):
            step_info.memory += important_contents + "\n"

        logger.info(f"🧠 All Memory: \n{step_info.memory}")

    @time_execution_async("--get_next_action")
    async def get_next_action(self, input_messages: list[BaseMessage], browserContext: Optional[CustomBrowserContext] = None, useOwnBrowser: Optional[bool] = False, enable_emunium = False) -> AgentOutput:
        """Get next action from LLM and insert cursor movement if targeting a new element."""
        fixed_input_messages = self._convert_input_messages(input_messages)
        """ print("Message", fixed_input_messages) """
        ai_message = self.llm.invoke(fixed_input_messages)

        self.message_manager._add_message_with_tokens(ai_message)
        
        
        print("Enable Emunium", enable_emunium)


        if hasattr(ai_message, "reasoning_content"):
            logger.info("🤯 Start Deep Thinking: ")
            logger.info(ai_message.reasoning_content)
            logger.info("🤯 End Deep Thinking")

        if isinstance(ai_message.content, list):
            ai_content = ai_message.content[0]
        else:
            ai_content = ai_message.content

        try:
            ai_content = ai_content.replace("```json", "").replace("```", "")
            ai_content = repair_json(ai_content)
            parsed_json = json.loads(ai_content)
            parsed: AgentOutput = self.AgentOutput(**parsed_json)
        except Exception as e:
            traceback.print_exc()
            logger.debug(ai_message.content)
            raise ValueError('Could not parse response.')

        if parsed is None:
            logger.debug(ai_message.content)
            raise ValueError('Could not parse response.')

        # Check for actions targeting elements and insert cursor movement if needed
        updated_actions = []
        if enable_emunium:
            for action in parsed.action:
                action_dict = action.model_dump(exclude_unset=True)

                # Extract selector and index, handling nested structures
                selector = None
                index = None
                action_name = next(iter(action_dict)) if action_dict else None
                if action_name and isinstance(action_dict.get(action_name), dict):
                    nested_dict = action_dict[action_name]
                    selector = nested_dict.get("selector")
                    index = nested_dict.get("index")

                target_identifier = None
                state = await self.browser_context.get_state()

                # Determine the target identifier (selector or index-based)
                if selector:
                    target_identifier = selector
                elif index is not None:
                    emunium = EmuniumPlaywright(self.browser_context)
                    page = await self.browser_context.get_current_page()

                    print("Page:", page)
                    print("Emunium:", emunium)
                    print("Index:", index)
                    print("Current Element:", state.selector_map[index])

                    # Extract the target element from selector_map using the index
                    target_element = state.selector_map[index]

                    # Initialize selector
                    element_selector = None

                    # Check if target_element is a DOMElementNode
                    if str(type(target_element).__name__) == 'DOMElementNode':
                        print("Target element is a DOMElementNode.")

                        # Try to access attributes from DOMElementNode
                        try:
                            element_id = target_element.get_attribute('id') if hasattr(target_element, 'get_attribute') else None
                            if element_id:
                                print("Extracted ElementId from DOMElementNode:", element_id)
                                element_selector = f'#{element_id}'
                        except Exception as e:
                            print(f"Failed to get attributes from DOMElementNode: {e}")

                        # Fallback: Use string representation if attributes are inaccessible
                        if not element_selector:
                            print("Falling back to string representation of DOMElementNode...")
                            element_str = str(target_element)

                            # Extract id
                            id_match = re.search(r'id\s*=\s*([\'"]?)([^\'"\s>]+)\1', element_str, re.IGNORECASE)
                            if id_match:
                                element_id = id_match.group(2)
                                print("Extracted ElementId from string:", element_id)
                                element_selector = f'#{element_id}'

                            # Extract class
                            if not element_selector:
                                class_match = re.search(r'class\s*=\s*([\'"]?)([^\'">]+)\1', element_str, re.IGNORECASE)
                                if class_match:
                                    class_name = class_match.group(2)
                                    tag_match = re.search(r'<(\w+)', element_str)
                                    tag_name = tag_match.group(1).lower() if tag_match else 'button'
                                    element_selector = f"{tag_name}.{'.'.join(class_name.split())}"
                                    print("Using class-based selector:", element_selector)

                            # Extract data-ved
                            if not element_selector:
                                data_ved_match = re.search(r'data-ved\s*=\s*([\'"]?)([^\'">]+)\1', element_str, re.IGNORECASE)
                                if data_ved_match:
                                    data_ved = data_ved_match.group(2)
                                    tag_match = re.search(r'<(\w+)', element_str)
                                    tag_name = tag_match.group(1).lower() if tag_match else 'button'
                                    element_selector = f'{tag_name}[data-ved="{data_ved}"]'
                                    print("Using data-ved selector:", element_selector)

                            # No usable attributes found
                            if not element_selector:
                                print("No usable attributes found in DOMElementNode string.")

                    # Handle other unexpected types

                    print("Constructed Selector:", element_selector)

                    # Locate the element using the constructed selector
                    TIMEOUT_MS = 30000

                    id_match = re.search(r'\bid\s*=\s*([\'"])(.*?)\1', element_str, re.IGNORECASE)
                    aria_match = re.search(r'aria-label\s*=\s*([\'"])(.*?)\1', element_str, re.IGNORECASE)
                    class_match = re.search(r'class\s*=\s*([\'"]?)([^\'">]+)\1', element_str, re.IGNORECASE)

                    if id_match:
                        element_id = id_match.group(2)
                        print("Extracted 2 from string:", element_id)
                        print("Self browser", browserContext)
                        print("Element Selector", element_selector)
                        element_selector = f'[id="{element_id}"]'
                        if browserContext:
                            await browserContext.move_to_element(element_selector, useOwnBrowser=useOwnBrowser)
                            
                    elif aria_match:
                        element_aria = aria_match.group(2)
                        print("Extracted aria-label from string:", element_aria)
                        print("Self browser", browserContext)
                        print("Element Selector", element_selector)
                        element_selector = f'[aria-label="{element_aria}"]'
                        if browserContext:
                            await browserContext.move_to_element(element_selector, useOwnBrowser=useOwnBrowser)

                    elif class_match:
                        element_id = class_match.group(2)
                        print("Extracted 3 from string:", element_id)
                        print("Self browser", browserContext)
                        print("Element Selector", element_selector)
                        element_selector = f'[class="{element_id}"]'
                        if browserContext:
                            await browserContext.move_to_element(element_selector, useOwnBrowser=useOwnBrowser)

                if target_identifier and target_identifier != self.last_cursor_selector:
                    # Insert a MoveCursorToElement action before the current action
                    move_action = {
                        "name": "Move cursor to element",
                        "selector": target_identifier
                    }
                    updated_actions.append(self.ActionModel(**move_action))
                    logger.debug(f"Inserted cursor movement to {target_identifier}")
                    self.last_cursor_selector = target_identifier

                    # Replace default actions with custom ones where applicable
                    if action_dict.get("name") == "input_text" and target_identifier:
                        updated_actions.append(self.ActionModel(
                            name="Type text at element",
                            selector=target_identifier,
                            text=action_dict["input_text"]["text"]
                        ))
                    elif action_dict.get("name") == "click_element" and target_identifier:
                        updated_actions.append(self.ActionModel(
                            name="Click element with human behavior",
                            selector=target_identifier
                        ))
                else:
                    updated_actions.append(action)
        else:
            # If enable_emunium is False, use the original actions without modification
            updated_actions = parsed.action

        parsed.action = updated_actions
        if len(parsed.action) > self.settings.max_actions_per_step:
           parsed.action = parsed.action[:self.settings.max_actions_per_step]
        self._log_response(parsed)
        return parsed
    
    @time_execution_async('--multi-act (agent)')
    async def multi_act_custom(
            self,
            actions: list[ActionModel],
            check_for_new_elements: bool = True,
        ) -> list[ActionResult]:
            """Execute multiple actions"""
            results = []
            
            print("555555")

            cached_selector_map = await self.browser_context.get_selector_map()
    
            cached_path_hashes = set(e.hash.branch_path_hash for e in cached_selector_map.values())

            await self.browser_context.remove_highlights()

            for i, action in enumerate(actions):
                if action.get_index() is not None and i != 0:
                    new_state = await self.browser_context.get_state()
                    new_path_hashes = set(e.hash.branch_path_hash for e in new_state.selector_map.values())
                    if check_for_new_elements and not new_path_hashes.issubset(cached_path_hashes):
                        # next action requires index but there are new elements on the page
                        msg = f'Something new appeared after action {i} / {len(actions)}'
                        logger.info(msg)
                        results.append(ActionResult(extracted_content=msg, include_in_memory=True))
                        break

                await self._raise_if_stopped_or_paused()

                result = await self.controller.act(
                    action,
                    self.browser_context,
                    self.settings.page_extraction_llm,
                    self.sensitive_data,
                    self.settings.available_file_paths,
                    context=self.context,
                )

                results.append(result)

                logger.debug(f'Executed action {i + 1} / {len(actions)}')
                if results[-1].is_done or results[-1].error or i == len(actions) - 1:
                    break

                await asyncio.sleep(self.browser_context.config.wait_between_actions)
                # hash all elements. if it is a subset of cached_state its fine - else break (new elements on page)

            return results   

    async def _run_planner(self) -> Optional[str]:
        """Run the planner to analyze state and suggest next steps"""
        # Skip planning if no planner_llm is set
        if not self.settings.planner_llm:
            return None

        # Create planner message history using full message history
        planner_messages = [
            PlannerPrompt(self.controller.registry.get_prompt_description()).get_system_message(),
            *self.message_manager.get_messages()[1:],  # Use full message history except the first
        ]

        if not self.settings.use_vision_for_planner and self.settings.use_vision:
            last_state_message: HumanMessage = planner_messages[-1]
            # remove image from last state message
            new_msg = ''
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg['type'] == 'text':
                        new_msg += msg['text']
                    elif msg['type'] == 'image_url':
                        continue
            else:
                new_msg = last_state_message.content

            planner_messages[-1] = HumanMessage(content=new_msg)

        # Get planner output
        response = await self.settings.planner_llm.ainvoke(planner_messages)
        plan = str(response.content)
        last_state_message = self.message_manager.get_messages()[-1]
        if isinstance(last_state_message, HumanMessage):
            # remove image from last state message
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg['type'] == 'text':
                        msg['text'] += f"\nPlanning Agent outputs plans:\n {plan}\n"
            else:
                last_state_message.content += f"\nPlanning Agent outputs plans:\n {plan}\n "

        try:
            plan_json = json.loads(plan.replace("```json", "").replace("```", ""))
            logger.info(f'📋 Plans:\n{json.dumps(plan_json, indent=4)}')

            if hasattr(response, "reasoning_content"):
                logger.info("🤯 Start Planning Deep Thinking: ")
                logger.info(response.reasoning_content)
                logger.info("🤯 End Planning Deep Thinking")

        except json.JSONDecodeError:
            logger.info(f'📋 Plans:\n{plan}')
        except Exception as e:
            logger.debug(f'Error parsing planning analysis: {e}')
            logger.info(f'📋 Plans: {plan}')
        return plan

    @time_execution_async("--step")
    async def step(self, step_info: Optional[CustomAgentStepInfo] = None, browserContext: Optional[CustomBrowserContext] = None, useOwnBrowser:  Optional[bool] = False, enable_emunium = False) -> None:
        """Execute one step of the task"""
        logger.info(f"\n📍 Step {self.state.n_steps}")
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_start_time = time.time()
        tokens = 0

        try:
            state = await self.browser_context.get_state()
            """ print("State", state) """


            await self._raise_if_stopped_or_paused()

            self.message_manager.add_state_message(state, self.state.last_action, self.state.last_result, step_info,
                                                   self.settings.use_vision)

            # Run planner at specified intervals if planner is configured
            if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
                await self._run_planner()
            input_messages = self.message_manager.get_messages()

            """ print("Input messages", input_messages) """
            tokens = self._message_manager.state.history.current_tokens

            """ print("Tokens", tokens) """

            try:
                model_output = await self.get_next_action(input_messages, browserContext = browserContext, useOwnBrowser = useOwnBrowser, enable_emunium = enable_emunium)
                self.update_step_info(model_output, step_info)
                self.state.n_steps += 1

                if self.register_new_step_callback:
                    await self.register_new_step_callback(state, model_output, self.state.n_steps)

                if self.settings.save_conversation_path:
                    target = self.settings.save_conversation_path + f'_{self.state.n_steps}.txt'
                    save_conversation(input_messages, model_output, target,
                                      self.settings.save_conversation_path_encoding)

                if self.model_name != "deepseek-reasoner":
                    # remove prev message
                    self.message_manager._remove_state_message_by_index(-1)
                await self._raise_if_stopped_or_paused()
            except Exception as e:
                # model call failed, remove last state message from history
                self.message_manager._remove_state_message_by_index(-1)
                raise e

            result: list[ActionResult] = await self.multi_act_custom(model_output.action)
            for ret_ in result:
                if ret_.extracted_content and "Extracted page" in ret_.extracted_content:
                    # record every extracted page
                    if ret_.extracted_content[:100] not in self.state.extracted_content:
                        self.state.extracted_content += ret_.extracted_content
            self.state.last_result = result
            self.state.last_action = model_output.action
            if len(result) > 0 and result[-1].is_done:
                if not self.state.extracted_content:
                    self.state.extracted_content = step_info.memory
                result[-1].extracted_content = self.state.extracted_content
                logger.info(f"📄 Result: {result[-1].extracted_content}")

            self.state.consecutive_failures = 0

        except InterruptedError:
            logger.debug('Agent paused')
            self.state.last_result = [
                ActionResult(
                    error='The agent was paused - now continuing actions might need to be repeated',
                    include_in_memory=True
                )
            ]
            return

        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result

        finally:
            step_end_time = time.time()
            actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps,
                    actions=actions,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=[r.error for r in result if r.error] if result else ['No result'],
                )
            )
            if not result:
                return

            if state:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)

    async def run(self, max_steps: int = 100, browserContext: Optional[CustomBrowserContext] = None, useOwnBrowser: Optional[bool] = False, enable_emunium=False) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""
        try:
            self._log_agent_run()

            print("Agent" , self)

            print("Browser Context" , browserContext)

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.multi_act_custom(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            step_info = CustomAgentStepInfo(
                task=self.task,
                add_infos=self.add_infos,
                step_number=1,
                max_steps=max_steps,
                memory="",
            )

            for step in range(max_steps):
                # Check if we should stop due to too many failures
                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    break

                # Check control flags before each step
                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
                    if self.state.stopped:  # Allow stopping while paused
                        break

                await self.step(step_info, browserContext = browserContext, useOwnBrowser = useOwnBrowser, enable_emunium = enable_emunium)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    break
            else:
                logger.info("❌ Failed to complete task in maximum steps")
                if not self.state.extracted_content:
                    self.state.history.history[-1].result[-1].extracted_content = step_info.memory
                else:
                    self.state.history.history[-1].result[-1].extracted_content = self.state.extracted_content

            return self.state.history

        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.state.agent_id,
                    is_done=self.state.history.is_done(),
                    success=self.state.history.is_successful(),
                    steps=self.state.n_steps,
                    max_steps_reached=self.state.n_steps >= max_steps,
                    errors=self.state.history.errors(),
                    total_input_tokens=self.state.history.total_input_tokens(),
                    total_duration_seconds=self.state.history.total_duration_seconds(),
                )
            )

            if not self.injected_browser_context:
                await self.browser_context.close()

            if not self.injected_browser and self.browser:
                await self.browser.close()

            if self.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif

                create_history_gif(task=self.task, history=self.state.history, output_path=output_path)
