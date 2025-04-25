from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Type
from datetime import datetime

from browser_use.agent.views import AgentOutput, AgentState, ActionResult, AgentHistoryList, MessageManagerState
from browser_use.controller.registry.views import ActionModel
from pydantic import BaseModel, ConfigDict, Field, create_model
from browser_use.browser.views import BrowserStateHistory
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from browser_use.agent.message_manager.views import MessageManagerState
from browser_use.agent.prompts import SystemPrompt
from browser_use.browser.views import BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.dom.history_tree_processor.service import (
	DOMElementNode,
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from browser_use.dom.views import SelectorMap



@dataclass
class BrowserStateHistoryCustom:
	url: str
	title: str
	screenshot: Optional[str] = None

	def to_dict(self) -> dict[str, Any]:
		data = {}
		data['screenshot'] = self.screenshot
		data['url'] = self.url
		data['title'] = self.title
		return data

@dataclass
class CustomAgentStepInfo:
    step_number: int
    max_steps: int
    task: str
    add_infos: str
    memory: str


class CustomAgentBrain(BaseModel):
    """Current state of the agent"""

    evaluation_previous_goal: str
    important_contents: str
    thought: str
    next_goal: str


class CustomAgentOutput(AgentOutput):
    """Output model for agent

    @dev note: this model is extended with custom actions in AgentService. You can also use some fields that are not in this model as provided by the linter, as long as they are registered in the DynamicActions model.
    """

    current_state: CustomAgentBrain

    @staticmethod
    def type_with_custom_actions(
            custom_actions: Type[ActionModel],
    ) -> Type["CustomAgentOutput"]:
        """Extend actions with custom actions"""
        model_ = create_model(
            "CustomAgentOutput",
            __base__=CustomAgentOutput,
            action=(
                list[custom_actions],
                Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
            ),  # Properly annotated field with no default
            __module__=CustomAgentOutput.__module__,
        )
        model_.__doc__ = 'AgentOutput model with custom actions'
        return model_


class CustomAgentState(BaseModel):
    agent_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%d%m%Y%H%M%S"),
        description="Agent ID formatted as DDMMYYYYHHMMSS"
    )
    n_steps: int = 1
    consecutive_failures: int = 0
    last_result: Optional[List['ActionResult']] = None
    history: AgentHistoryList = Field(default_factory=lambda: AgentHistoryList(history=[]))
    last_plan: Optional[str] = None
    paused: bool = False
    stopped: bool = False

    message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState)

    last_action: Optional[List['ActionModel']] = None
    extracted_content: str = ''
    
    
    
class StepMetadata(BaseModel):
	"""Metadata for a single step including timing and token information"""

	step_start_time: float
	step_end_time: float
	input_tokens: int  # Approximate tokens from message manager for this step
	step_number: int

	@property
	def duration_seconds(self) -> float:
		"""Calculate step duration in seconds"""
		return self.step_end_time - self.step_start_time   
    

class AgentHistoryCustom(BaseModel):
	"""History item for agent actions"""

	model_output: AgentOutput | None
	result: list[ActionResult]
	state: BrowserStateHistoryCustom
	metadata: Optional[StepMetadata] = None

	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

	@staticmethod
	def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
		elements = []
		for action in model_output.action:
			index = action.get_index()
			if index and index in selector_map:
				el: DOMElementNode = selector_map[index]
				elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
			else:
				elements.append(None)
		return elements

	def model_dump(self, **kwargs) -> Dict[str, Any]:
		"""Custom serialization handling circular references"""

		# Handle action serialization
		model_output_dump = None
		if self.model_output:
			action_dump = [action.model_dump(exclude_none=True) for action in self.model_output.action]
			model_output_dump = {
				'current_state': self.model_output.current_state.model_dump(),
				'action': action_dump,  # This preserves the actual action data
			}

		return {
			'model_output': model_output_dump,
			'result': [r.model_dump(exclude_none=True) for r in self.result],
			'state': self.state.to_dict(),
		}
