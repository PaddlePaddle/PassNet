from typing import Dict, Callable
from dataclasses import dataclass
from typing import Dict, Any
from abc import ABC, abstractmethod
from ai4c.agent.prompt.prompt_manager import PromptManager
from ai4c.utils.llm_query_utils import (
    LLMQueryConfig,
    query_llm_service,
    add_token_usage
)


@dataclass
class AgentMessage:
    sender: str         # which agent is the current message from
    content: str        # The natural language reasoning or prompt
    code_content: str   # Pass Code
    runtime_info: str   # e.g., error messages, stdout logs
    token_usage: Dict[str, Any]
    is_terminal: bool = False
    
    def update_token_usage(self, new_usage):
        self.token_usage = add_token_usage(self.token_usage, new_usage)


class LLMClient:
    def __init__(self, llm_query_config: LLMQueryConfig):
        self.config = llm_query_config
        self._service_func = query_llm_service(llm_query_config)

    def chat(self, user_prompt: str, system_prompt: str):
        result = self._service_func(user_prompt, system_prompt)
        
        if result is None:
            raise RuntimeError("LLM Service returned None (Max retries exceeded).")
            
        return result


class BaseAgent(ABC):
    def __init__(
        self, 
        name: str, 
        llm_config: LLMQueryConfig, 
        template_dir: str,
        system_prompt: str = "You are a help assistant.",
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.prompt_manager = PromptManager(template_dir)
        self.client = LLMClient(llm_config)
        self.history = [{"role": "system", "content": system_prompt}]
        self.tools: Dict[str, Callable] = {}

    @abstractmethod
    def process(self, msg: list[AgentMessage]) -> list[AgentMessage]:
        pass

    def register_tool(self, func: Callable):
        self.tools[func.__name__] = func

    def render_prompt(self, template_name: str, **kwargs):
        return self.prompt_manager.render(template_name, **kwargs)

    def _set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": system_prompt}]
        
    def _format_history_to_prompt(self, new_content):
        formatted_prompt = ""
        for msg in self.history:
            formatted_prompt += f"<{msg['role']}>\n{msg['content']}\n</{msg['role']}>\n"
        if new_content:
            formatted_prompt += f"<user>\n{new_content}\n</user>\n"
        return formatted_prompt
