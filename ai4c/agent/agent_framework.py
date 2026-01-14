from typing import Dict
from ai4c.agent.agent_base import AgentMessage, BaseAgent


class AgentWorkflowEngine:
    def __init__(self, max_turns: int = 10):
        self._agents: Dict[str, BaseAgent] = {}
        self._transitions: Dict[str, str] = {}
        self._max_turns = max_turns
        self._first_agent_name = None

    def register_agent(self, agent: BaseAgent):
        self._agents[agent.name] = agent
        return self

    def set_first_agent(self, agent_name):
        self._first_agent_name = agent_name

    def register_transition(self, source_agent_name: str, target_agent_name: str):
        self._transitions[source_agent_name] = target_agent_name

    def run(self, initial_task_message: AgentMessage):
        assert len(self._agents) > 0, "There is no agent registered."
        assert self._first_agent_name is not None, "First agent not set."

        response_msg = [initial_task_message]
        agent_name = self._first_agent_name

        print("[Agent Framework] Start Working")

        for turn in range(self._max_turns):
            response_msg = self._run_agent_turn(agent_name, response_msg, turn_num=turn)

    def _run_agent_turn(self, agent_name, response_msg, turn_num):
        """Run agent workflow until no next agent is found."""

        while agent_name is not None:
            print(f"[Agent Framework] iter: {turn_num}, '{agent_name}' is working")
            response_msg = self._execute_agent(agent_name, response_msg)
            agent_name = self._get_next_agent_name(agent_name)
        return response_msg

    def _execute_agent(self, agent_name, msg: list[AgentMessage]):
        current_agent_instance = self._agents[agent_name]
        resp_msg = current_agent_instance.process(msg)
        return resp_msg

    def _get_next_agent_name(self, current_agent_name):
        return self._transitions.get(current_agent_name, None)
