from typing import Dict
from ai4c.agent.agent_base import (
    BaseAgent,
    AgentMessage
)


class AgentWorkflowEngine:
    def __init__(self, max_turns: int = 10):
        self._agents: Dict[str, BaseAgent] = {}
        self._transitions: Dict[str, str] = {}
        self.max_turns = max_turns

    def register_agent(self, agent: BaseAgent):
        self._agents[agent.name] = agent

    def register_transition(self, source_agent: str, target_agent: str):
        self._transitions[source_agent] = target_agent

    def run(self, initial_msg: AgentMessage) -> AgentMessage:
        pass
    
    def _get_next_receiver(self, current_msg: AgentMessage) -> str:
        pass
    