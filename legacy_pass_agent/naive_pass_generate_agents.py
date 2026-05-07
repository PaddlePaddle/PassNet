import argparse
from legacy_pass_agent.agent.agent_base import LLMQueryConfig, AgentMessage
from legacy_pass_agent.agent.agent_framework import AgentWorkflowEngine
from legacy_pass_agent.agent.instance_agent_analysis import AnalysisAgent
from legacy_pass_agent.agent.instance_agent_engineer import EngineerAgent


def construct_init_message(args):
    init_message = AgentMessage(
        sender="<initial>",
        content=None,
        code_content=None,
        meta_info={},
        token_usage=None,
        is_terminal=False,
    )
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
        system_prompt=None,
    )
    analysis_agent.set_system_prompt(
        analysis_agent.render_prompt("system_analysis_agent.j2")
    )
    engineer_agent = EngineerAgent(
        name="EngineerAgent",
        llm_config=llm_query_config,
        template_dir=args.template_dir,
        system_prompt=None,
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
    agent_framework.register_transition("AnalysisAgent", "EngineerAgent")
    initial_message = construct_init_message(args)
    agent_framework.run(initial_message)


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
        default="legacy_pass_agent/agent/prompt/templates",
        help="The path of prompt template dir.",
    )
    parser.add_argument(
        "--dsl",
        type=str,
        required=False,
        default=None,
        help="The domain specific language for code generation in pass optimization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="The device which profile runs on",
    )
    args = parser.parse_args()
    main(args)
