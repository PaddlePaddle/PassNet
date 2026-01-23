from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any
from ai4c.agent.agent_base import AgentMessage, BaseAgent


@dataclass
class TurnContext:
    turn_idx: int
    max_turns: int
    initial_message: AgentMessage
    final_message: Optional[AgentMessage] = None
    last_eval: Optional[Any] = None


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

    def _run_agent_turn(self, initial_message: AgentMessage, turn_num: int) -> AgentMessage:
        assert self._first_agent_name is not None, "First agent not set."
        agent_name = self._first_agent_name
        response_msg: list[AgentMessage] = [initial_message]

        while agent_name is not None:
            print(f"[Agent Framework] iter: {turn_num}, '{agent_name}' is working")
            response_msg = self._execute_agent(agent_name, response_msg)
            agent_name = self._get_next_agent_name(agent_name)

        if isinstance(response_msg, list) and response_msg:
            return response_msg[0]
        raise RuntimeError("Agent chain returned no messages.")

    def _execute_agent(self, agent_name, msg: list[AgentMessage]) -> list[AgentMessage]:
        current_agent_instance = self._agents[agent_name]
        resp_msg = current_agent_instance.process(msg)
        return resp_msg

    def _get_next_agent_name(self, current_agent_name):
        return self._transitions.get(current_agent_name, None)

    def run_multi_round(
        self,
        *,
        message_factory: Callable[[int], AgentMessage],
        on_after_round: Optional[Callable[[TurnContext], Any]] = None,
        on_before_round: Optional[Callable[[TurnContext], None]] = None,
    ):
        assert len(self._agents) > 0, "There is no agent registered."
        assert self._first_agent_name is not None, "First agent not set."

        last_eval = None
        last_final = None

        for turn in range(self._max_turns):
            init_msg = message_factory(turn)
            ctx = TurnContext(
                turn_idx=turn,
                max_turns=self._max_turns,
                initial_message=init_msg,
                final_message=last_final,
                last_eval=last_eval,
            )
            if on_before_round is not None:
                on_before_round(ctx)

            print(f"[Agent Framework] Start round {turn+1}/{self._max_turns}")
            last_final = self._run_agent_turn(ctx.initial_message, turn_num=turn)
            ctx.final_message = last_final

            if on_after_round is not None:
                last_eval = on_after_round(ctx)
                ctx.last_eval = last_eval

                # Early exit: if evaluation succeeded, we're done.
                try:
                    if getattr(last_eval, "status", None) == "success":
                        print(f"[Agent Framework] Early exit on success at round {turn+1}/{self._max_turns}")
                        break
                except Exception:
                    pass

        return last_eval
