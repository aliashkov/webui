from __future__ import annotations

import logging
import pdb
from typing import List, Optional, Type, Dict

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, AgentStepInfo, ActionModel
from browser_use.browser.views import BrowserState
from browser_use.agent.message_manager.service import MessageManagerSettings
from browser_use.agent.views import ActionResult, AgentOutput, AgentStepInfo, MessageManagerState
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI
from ..utils.llm import DeepSeekR1ChatOpenAI
from .custom_prompts import CustomAgentMessagePrompt

# Убедитесь, что эти импорты есть, если они используются в типах или значениях по умолчанию
from browser_use.agent.views import AgentSettings
from browser_use.agent.views import AgentStepInfo
from browser_use.agent.prompts import AgentMessagePrompt # Для типа в CustomMessageManagerSettings

logger = logging.getLogger(__name__)


class CustomMessageManagerSettings(MessageManagerSettings):
    agent_prompt_class: Type[AgentMessagePrompt] = AgentMessagePrompt


class CustomMessageManager(MessageManager):
    def __init__(
            self,
            task: str,
            system_message: SystemMessage,
            agent_settings: AgentSettings,
            settings: CustomMessageManagerSettings = CustomMessageManagerSettings(),
            state: MessageManagerState = MessageManagerState(),
    ):
        super().__init__(
            task=task,
            system_message=system_message,
            settings=settings,
            state=state
        )
        self.main_agent_settings: AgentSettings = agent_settings
        self.system_prompt: SystemMessage = system_message
        # _init_messages, вероятно, вызывается в super().__init__ или должен быть вызван здесь,
        # если вы переопределяете его и базовая версия не вызывается.
        # Если _init_messages из MessageManager не вызывается или вы его переопределили без вызова super:
        # self._init_messages()


    def _init_messages(self) -> None:
        """Initialize the message history with system message, context, task, and other initial messages"""
        self._add_message_with_tokens(self.system_prompt)
        self.context_content = ""

        if self.settings.message_context:
            self.context_content += 'Context for the task' + self.settings.message_context

        if self.settings.sensitive_data:
            info = f'Here are placeholders for sensitive data: {list(self.settings.sensitive_data.keys())}'
            info += 'To use them, write <secret>the placeholder name</secret>'
            self.context_content += info

        if self.settings.available_file_paths:
            filepaths_msg = f'Here are file paths you can use: {self.settings.available_file_paths}'
            self.context_content += filepaths_msg

        if self.context_content:
            context_message = HumanMessage(content=self.context_content)
            self._add_message_with_tokens(context_message)

    def cut_messages(self):
        """Get current message list, potentially trimmed to max tokens"""
        diff = self.state.history.current_tokens - self.settings.max_input_tokens
        min_message_len = 2 if self.context_content is not None else 1

        while diff > 0 and len(self.state.history.messages) > min_message_len:
            self.state.history.remove_message(min_message_len)  # always remove the oldest message
            diff = self.state.history.current_tokens - self.settings.max_input_tokens

    def add_state_message(
            self,
            state: BrowserState,
            actions: Optional[List[ActionModel]] = None, # We know this isn't in the provided __init__
            result: Optional[List[ActionResult]] = None,
            step_info: Optional[AgentStepInfo] = None,
            use_vision=True,
    ) -> None:
        logger.info(f"DEBUG: Instantiating {self.settings.agent_prompt_class.__name__}")
        logger.info(f"DEBUG: include_attributes to be passed: {self.settings.include_attributes}")

        try:
            
            print("Actions", actions)
            
            # Call with ONLY the arguments defined in the AgentMessagePrompt code you provided
            prompt_instance = self.settings.agent_prompt_class(
                state=state,
                # actions=actions, # Temporarily remove this, as it's not in the __init__
                result=result,
                include_attributes=self.settings.include_attributes, # This is a defined kwarg
                step_info=step_info
            )
            
            """ print("Prompt instance", prompt_instance) """
            
            state_message = prompt_instance.get_user_message(use_vision)
            
            """ print("State message", state_message) """
            
            self._add_message_with_tokens(state_message)
        except TypeError as e:
            logger.error(f"DEBUG: TypeError during AgentMessagePrompt instantiation: {e}")
            import inspect
            sig = inspect.signature(self.settings.agent_prompt_class.__init__)
            logger.error(f"DEBUG: Effective signature of {self.settings.agent_prompt_class.__name__}.__init__: {sig}")
            # Log the actual arguments being passed if we reach here again
            passed_args_tuple = (state, result, self.settings.include_attributes, step_info) # based on current call
            logger.error(f"DEBUG: Args tuple if calling positionally: {passed_args_tuple}")
            raise
    
    def _remove_state_message_by_index(self, remove_ind=-1) -> None:
        """Remove state message by index from history"""
        i = len(self.state.history.messages) - 1
        remove_cnt = 0
        while i >= 0:
            if isinstance(self.state.history.messages[i].message, HumanMessage):
                remove_cnt += 1
            if remove_cnt == abs(remove_ind):
                self.state.history.messages.pop(i)
                break
            i -= 1
