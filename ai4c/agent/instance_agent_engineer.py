import json
from ai4c.agent.agent_base import BaseAgent
from ai4c.utils.common_string_utils import extract_code_blocks

register_reference = {
    "triton": ["agent_analysis_pass.j2"]
}


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
            
            code = self._process_one_pass(pass_info)
            generated_codes[pass_name] = code
            



    def _process_one_msg(self, pass_info, dsl, backend):
        references = _get_references(dsl)
        references_content = []
        for ref in references:
            references_content.append(self.render_prompt(ref))
        
        optimization_prompt = self.render_prompt(
            "engineer_user.j2", 
            pass_info=pass_info,
            dsl=dsl,
            backend=backend,
            references=references_content
        )

        response = self.client.chat(user_prompt=optimization_prompt, system_prompt=self.system_prompt)
        code = extract_code_blocks(response, ["python", ""])
        return code

    def _handle_init_message(self, messages):
        ''' fetch DSL and device setting from meta_info ''' 
        
        return {
            "dsl": messages.meta_info["dsl"],
            "device": messages.meta_info["device"]
        }