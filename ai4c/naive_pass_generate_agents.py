import argparse
from ai4c.agent.agent_base import LLMQueryConfig, AgentMessage
from ai4c.agent.agent_framework import AgentWorkflowEngine
from ai4c.agent.instance_agent_analysis import AnalysisAgent
from ai4c.agent.instance_agent_engineer import EngineerAgent


def construct_init_message(args):
    init_message = AgentMessage()
    init_message.sender = "<initial>"
    init_message.meta_info["task_path"] = args.model_dir
    init_message.meta_info["dsl"] = args.dsl
    init_message.meta_info["device"] = args.device
    return init_message


def main(args):
    llm_query_config = LLMQueryConfig()
    agent_framework = AgentWorkflowEngine(args.max_turns)

    # custom agent framework
    analysis_agent = AnalysisAgent(
        name="AnalysisAgent",
        llm_config=llm_query_config,
        template_dir=args.template_dir,
    )
    analysis_agent.set_system_prompt(
        analysis_agent.render_prompt("system_analysis_agent.j2")
    )
    engineer_agent = EngineerAgent(
        name="EngineerAgent",
        llm_config=llm_query_config,
        template_dir=args.template_dir,
    )
    engineer_agent.set_system_prompt(
        engineer_agent.render_prompt(
            "system_engineer_agent.j2",
            dsl=args.dsl,
            backend=args.device,
        )
    )

    # registe agents and run
    agent_framework.register_agent(analysis_agent).register_agent(engineer_agent)
    agent_framework.set_first_agent("AnalysisAgent")
    agent_framework.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, required=False, default=None, help="the task dir"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        required=False,
        default=1,
        help="the max turns that agent framework can run",
    )
    parser.add_argument(
        "--template-dir",
        type=str,
        required=False,
        default="ai4c/agent/prompt/templates",
        help="The path of prompt template dir.",
    )
    parser.add_argument(
        "--dsl",
        type=str,
        required=False,
        default=None,
        help="The domain specific language for code generation in pass optimization."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="The device which profile runs on"
    )
