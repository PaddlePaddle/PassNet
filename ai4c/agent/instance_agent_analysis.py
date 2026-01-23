import os
import json
from ai4c.agent.agent_base import BaseAgent, AgentMessage
from ai4c.utils.common_string_utils import read_file, extract_code_blocks


class AnalysisAgent(BaseAgent):

    def __init__(self, name, llm_config, template_dir, system_prompt):
        super().__init__(name, llm_config, template_dir, system_prompt)

    def process(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        init_message = messages[0]
        requirement = self._handle_init_message(init_message)
        # Optional multi-turn feedback from the previous evaluation round.
        requirement["last_run_feedback"] = (init_message.meta_info or {}).get(
            "last_run_feedback"
        )
        requirement["last_run_status"] = (init_message.meta_info or {}).get("last_run_status")
        requirement["last_run_rectified_speedup"] = (init_message.meta_info or {}).get(
            "last_run_rectified_speedup"
        )
        requirement["last_run_speedup_gpu"] = (init_message.meta_info or {}).get(
            "last_run_speedup_gpu"
        )
        requirement["last_pass_artifacts"] = (init_message.meta_info or {}).get(
            "last_pass_artifacts"
        )
        requirement["fixed_pass_names"] = (init_message.meta_info or {}).get(
            "fixed_pass_names"
        )

        analysis_prompt = self.render_prompt("agent_analysis_pass.j2", **requirement)
        query_result = self.client.chat(
            user_prompt=analysis_prompt, system_prompt=self.system_prompt
        )
        response_text = query_result.response_text
        token_usage = query_result.token_usage

        pass_plan = extract_code_blocks(response_text, ["json", ""])
        new_msg = AgentMessage(
            sender=self.name,
            content=response_text,
            code_content=json.dumps(pass_plan, ensure_ascii=False),
            meta_info=init_message.meta_info,
            token_usage={},
            is_terminal=False,
        )
        new_msg.update_token_usage(token_usage)
        return [new_msg]

    def _handle_init_message(self, msg: AgentMessage):
        """analyze the task requirement"""

        requirement_info = {"entry": None, "graphs_data": []}
        task_path = msg.meta_info["task_path"]

        # read the requirement file
        graph_file_path = read_file(os.path.join(task_path, "graph_list.txt")).split(
            "\n"
        )[:-1]

        entry_info = read_file(os.path.join(task_path, "entry.sh"))
        requirement_info["entry"] = entry_info
        for gf in graph_file_path:
            graph_data = {"name": gf, "model_code": None, "weight_meta": None}
            absolute_graph_path = os.path.join(task_path, gf)
            model_code = read_file(os.path.join(absolute_graph_path, "model.py"))
            weight_meta = read_file(os.path.join(absolute_graph_path, "weight_meta.py"))
            graph_data["model_code"] = model_code
            graph_data["weight_meta"] = weight_meta
            requirement_info["graphs_data"].append(graph_data)

        return requirement_info

    def _handle_profile_message(self, msg):
        pass
