import os
import json
from ai4c.agent.agent_base import BaseAgent, AgentMessage
from ai4c.utils.common_string_utils import extract_last_code_block, write_file

register_reference = {"triton": ["agent_analysis_pass.j2"]}


def _get_references(dsl_type):
    return register_reference.get(dsl_type, [])


class EngineerAgent(BaseAgent):

    def __init__(self, name, llm_config, template_dir, system_prompt):
        super().__init__(name, llm_config, template_dir, system_prompt)

    def process(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        init_message = messages[0]
        meta_info = self._handle_init_message(init_message)

        pass_plan_json_str = json.loads(init_message.code_content)
        pass_plan = json.loads(pass_plan_json_str)

        fixed_names_env = os.getenv("AI4C_FIXED_PASS_NAMES")
        if fixed_names_env:
            try:
                fixed_names = json.loads(fixed_names_env)
            except Exception:
                fixed_names = None
            if isinstance(fixed_names, list) and fixed_names:
                pass_details = pass_plan.get("pass_details", [])
                if isinstance(pass_details, list):
                    for i, p in enumerate(pass_details):
                        if i < len(fixed_names) and isinstance(p, dict):
                            p["name"] = fixed_names[i]
                # Keep pass_order consistent with fixed names (trim to available passes)
                pass_plan["pass_order"] = fixed_names[: len(pass_details)]

        pass_details = pass_plan.get("pass_details", [])
        for pass_info in pass_details:
            pass_name = pass_info.get("name")
            print(f"[{self.name}] Implementing {pass_name} using {meta_info['dsl']}...")

            optimized_pass_code, token_usage = self._process_one_pass(
                pass_info, meta_info["dsl"], meta_info["device"]
            )
            pass_info["optimized_pass_code"] = optimized_pass_code
            init_message.update_token_usage(token_usage)

        # dump the optimized pass plan
        self._dump_pass_plan(meta_info["task_path"], pass_plan)

        new_msg = AgentMessage(
            sender=self.name,
            content="",
            code_content=json.dumps(pass_plan, ensure_ascii=False),
            meta_info=init_message.meta_info,
            token_usage=init_message.token_usage,
            is_terminal=False,
        )
        return [new_msg]

    def _process_one_pass(self, pass_info, dsl, backend):
        references = _get_references(dsl)
        references_content = []
        for ref in references:
            references_content.append(self.render_prompt(ref))

        pass_code = pass_info.get("pass_code", "")
        optimization_prompt = self.render_prompt(
            "agent_engineer_pass.j2",
            pass_info=pass_info,
            pass_code=pass_code,
            dsl=dsl,
            backend=backend,
            references=references_content,
            last_run_feedback=(os.environ.get("AI4C_LAST_RUN_FEEDBACK") or ""),
            last_pass_artifacts=(os.environ.get("AI4C_LAST_PASS_ARTIFACTS") or ""),
        )

        response = self.client.chat(
            user_prompt=optimization_prompt, system_prompt=self.system_prompt
        )
        response_text = response.response_text
        token_usage = response.token_usage

        # Take only the last code block to avoid accidental concatenation of
        # multiple blocks (which can make pass files grow across rounds).
        code = extract_last_code_block(response_text, ["python", ""])
        return code, token_usage

    def _handle_init_message(self, messages):
        """fetch DSL and device setting from meta_info"""
        return {
            "task_path": messages.meta_info["task_path"],
            "dsl": messages.meta_info["dsl"],
            "device": messages.meta_info["device"],
        }

    def _dump_pass_plan(self, task_path, pass_plan):
        workdir_path = f"{task_path}/pass_dir"

        # write sorted_output_pass_rule_names.json
        pass_order = pass_plan["pass_order"]
        sorted_output_pass_rule_path = os.path.join(
            workdir_path, "sorted_output_pass_rule_names.json"
        )
        write_file(
            sorted_output_pass_rule_path, json.dumps(pass_order, ensure_ascii=False)
        )

        # write pass file
        passes = pass_plan["pass_details"]
        for _pass in passes:
            pass_name = _pass["name"]
            pass_code = _pass.get("optimized_pass_code", "")
            pass_file_path = os.path.join(workdir_path, f"{pass_name}.py")
            write_file(pass_file_path, pass_code)
