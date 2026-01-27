import argparse
from functools import partial

from ai4c.utils.eval_runner import prepare_round, evaluate_round
from ai4c.agent.agent_base import LLMQueryConfig, AgentMessage
from ai4c.agent.agent_framework import AgentWorkflowEngine
from ai4c.agent.instance_agent_analysis import AnalysisAgent
from ai4c.agent.instance_agent_engineer import EngineerAgent


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
    engine = AgentWorkflowEngine(args.max_turns)

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

    engine.register_agent(analysis_agent).register_agent(engineer_agent)
    engine.set_first_agent("AnalysisAgent")
    engine.register_transition("AnalysisAgent", "EngineerAgent")

    def message_factory(_turn_idx):
        return construct_init_message(args)

    evaluate_round_bound = partial(evaluate_round, eval_output_dir=args.eval_output_dir)

    engine.run_multi_round(
        message_factory=message_factory,
        prepare_round=prepare_round,
        evaluate_round=evaluate_round_bound,
    )


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
        help="number of multi-round iterations. Each iteration generates passes, runs entry.sh, then feeds validation.log feedback into the next iteration.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing validation.log and aggregated_score.json produced by entry.sh. Required when --max-turns > 1.",
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
    if args.max_turns > 1 and not args.eval_output_dir:
        parser.error("--eval-output-dir is required when --max-turns > 1")
    main(args)
