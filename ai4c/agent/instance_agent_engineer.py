import os
import json
import uuid
from ai4c.agent.agent_base import BaseAgent, AgentMessage
from ai4c.utils.common_string_utils import extract_code_blocks, write_file

register_reference = {"triton": ["agent_analysis_pass.j2"]}


def _get_references(dsl_type):
    return register_reference.get(dsl_type, [])


class EngineerAgent(BaseAgent):

    def __init__(self, name, llm_config, template_dir, system_prompt):
        super().__init__(name, llm_config, template_dir, system_prompt)

    def process(self, messages):
        init_message = messages[-1]
        meta_info = self._handle_init_message(init_message)
        pass_plan = json.loads(init_message.code_content)

        pass_details = pass_plan.get("pass_details", [])
        for pass_info in pass_details:
            pass_name = pass_info.get("name")
            print(f"[{self.name}] Implementing {pass_name} using {meta_info['dsl']}...")

            code, token_usage = self._process_one_pass(pass_info, meta_info["dsl"], meta_info["device"])
            pass_info["optimized_pass_code"] = code
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
        return new_msg

    def _process_one_pass(self, pass_info, dsl, backend):
        references = _get_references(dsl)
        references_content = []
        for ref in references:
            references_content.append(self.render_prompt(ref))

        optimization_prompt = self.render_prompt(
            "engineer_user.j2",
            pass_info=pass_info,
            dsl=dsl,
            backend=backend,
            references=references_content,
        )

        response = self.client.chat(
            user_prompt=optimization_prompt, system_prompt=self.system_prompt
        )
        response_text = response.response_text
        token_usage = response.token_usage
        
        code = extract_code_blocks(response_text, ["python", ""])
        return code, token_usage

    def _handle_init_message(self, messages):
        """fetch DSL and device setting from meta_info"""
        return {
            "task_path": messages.meta_info["task_path"],
            "dsl": messages.meta_info["dsl"],
            "device": messages.meta_info["device"],
        }

    def _dump_pass_plan(self, task_path, pass_plan):
        plan_id = str(uuid.uuid4())[:8]
        workdir_path = f"{task_path}/pass_{plan_id}"

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
