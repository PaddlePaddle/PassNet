"""
AI4C Agent - extends R2E-Gym Agent with AI4C-specific tool support.

This agent always uses function calling (hardcoded) and only implements
the minimal logic needed for AI4C tasks.
"""

import time
import copy
import json
import traceback
from typing import List, Dict, Any
import re

import litellm
from r2egym.agenthub.agent.agent import Agent as R2EGymAgent
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory


class AI4CAgent(R2EGymAgent):
    """
    AI4C Agent that extends R2E-Gym Agent with:
    - AI4C-specific tools (file_editor, pass_evaluator)
    - Hardcoded use_fn_calling=True (always uses function calling)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hardcode use_fn_calling to True
        self.use_fn_calling = True
        self.max_score = -1.0
    
    def extract_speedup(self, observation: str) -> float:
        """Return the speedup value embedded in an observation string.

        Tries several patterns in priority order; returns 0.0 on failure.
        """
        if not observation:
            return 0.0
        patterns = [
            r'rectified_speedup\s*=\s*([\d.e+\-]+)',
            r'"score"\s*:\s*([\d.e+\-]+)',
            r'Speedup[:\s]+([\d.e+\-]+)',
        ]
        for pat in patterns:
            m = re.search(pat, observation)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return 0.0
    
    def parse_output_patch(self, env, pass_dir_output: str) -> str:
        output_patch = ""
        try:
            # Try to read the pass files if they exist
            if pass_dir_output.strip():
                # output_patch = f"Pass files created:\n{pass_dir_output}"
                output_patch_files = pass_dir_output.strip().split("\n")
                for idx, file in enumerate(output_patch_files):
                    output_patch += "-"*20 + f" {file} " + "-"*20
                    file_content, _ = env.runtime.run(f"cat {file.strip()}", timeout=10)
                    output_patch += f"\n{file_content}\n\n"
                output_patch += "-"*20 + f" pass_dir/score.txt " + "-"*20
                output_patch += f"\n{self.max_score}\n\n"
        except:
            pass
        return output_patch
    
    def update_speedup_and_pass_status(self, env, observation: str) -> str:
        score = self.extract_speedup(observation)
        if score > self.max_score:
            self.max_score = score
            try:
                pass_dir_output, _ = env.runtime.run("ls pass_dir/*.py pass_dir/*.json 2>/dev/null || echo ''", timeout=10)
            except:
                pass
        return pass_dir_output

    def run(
        self,
        env,
        use_fn_calling: bool = True,  # Ignored - always True
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        max_token_limit: int = 65536,
        max_exec_time: int = 90,
        max_total_time: int = 50000,
        max_llm_time: int = 7200,
        temperature: float = 0,
        metadata: Dict[str, Any] = None,
        scaffold: str = "r2egym",
    ):
        """
        Simplified run() with hardcoded function calling.
        """
        if metadata is None:
            metadata = {}

        # Hardcode use_fn_calling
        self.use_fn_calling = True
        self.llm_timeout = max_llm_time
        self.logger.warning(f"Using fn calling: {self.use_fn_calling} (AI4C hardcoded)")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        # Prepare problem_statement from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
        gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)

        # Get system and instance prompts
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir='/testbed',
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )

        # Initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Initialize the parameters
        done = False
        step_count = 0
        total_time_traj = 0
        pass_dir_output = ""
        self.trajectory_steps: List[TrajectoryStep] = []

        # Agent loop
        while not done:
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1]["content"] += f"\n{stepcount_message}"
            self.logger.info(stepcount_message)

            # Query the LLM
            messages = copy.deepcopy(self.history)
            try:
                response, llm_exec_time = self.model_query(messages, temperature)
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                done = True
                break

            # Log total tokens
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
                self.logger.info(f"Total Tokens: {total_tokens}")
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = self._count_tokens(messages)

            # Parse the LLM response using custom_parser (for function calling)
            thought, action = self.custom_parser(response)

            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action.to_bashcmd()}\n")

            # Send the action to the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
                if action.function_name == "pass_evaluator":
                    pass_dir_output = self.update_speedup_and_pass_status(env, obs.bash_output)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1

            # Update history (function calling mode)
            assistant_response = response.choices[0].message.dict()
            if assistant_response.get("tool_calls", None):
                assistant_response["tool_calls"] = assistant_response["tool_calls"][:1]
            self.history.append(assistant_response)

            # Add tool response to history
            try:
                function_name = response.choices[0].message.tool_calls[0].function.name
                function_id = response.choices[0].message.tool_calls[0].id
                self.history.append({
                    "role": "tool",
                    "content": str(obs),
                    "name": function_name,
                    "tool_call_id": function_id,
                })
            except Exception as e:
                self.logger.error(f"Error logging tool response: {e}")
                self.history.append({"role": "user", "content": str(obs)})

            # Log the observation
            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)

            # Check limits
            if done:
                self.logger.info(f"Agent finished. Steps: {step_count}")
                exit_reason = "agent"
            elif total_tokens >= max_token_limit:
                self.logger.info(f"Token limit reached: {total_tokens}")
                done = True
                exit_reason = "token_limit"
            elif step_count >= max_steps_absolute:
                self.logger.info(f"Max steps reached: {step_count}")
                done = True
                exit_reason = "abs_step_limit"
            elif total_time_traj >= max_total_time:
                self.logger.info(f"Max time reached: {total_time_traj}")
                done = True
                exit_reason = "traj_time_limit"
            else:
                exit_reason = None

            # Store trajectory step with correct fields
            traj_step = TrajectoryStep(
                step_idx=step_count,
                thought=thought,
                action=action.to_xml_string(),  # Convert Action to string
                observation=str(obs),  # Convert Observation to string
                done=done,
                info=info,
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
            )
            self.trajectory_steps.append(traj_step)

        self.logger.info(f"Agent run complete. Total steps: {step_count}")

        # Get output patch (for AI4C, this would be the pass files)
        output_patch = self.parse_output_patch(env, pass_dir_output)
        
        # Create Trajectory object
        trajectory = Trajectory(
            trajectory_steps=self.trajectory_steps,
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            exp_name=metadata.get("exp_name", "ai4c"),
            env_args={},
            agent_args={},
            ds=env.runtime.ds,
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason if exit_reason else "unknown",
            output_patch=output_patch,
        )

        self.logger.info(f"Returning trajectory with exit_reason: {trajectory.exit_reason}")
        self.logger.info(f"Return type: Trajectory object (open-source r2e-gym expects single return)")

        # Store history as instance variable so it can be accessed later
        self.final_history = self.history

        # Open-source r2e-gym expects agent.run() to return just Trajectory, not tuple
        return trajectory

    def model_query(self, messages: List[Dict[str, str]], temperature: float = 0) -> Dict[str, Any]:
        """
        Model query with AI4C tools (always uses function calling).
        """
        from r2egym.agenthub.tools import file_editor

        # Define pass_evaluator tool (matching the original R2E-Gym definition)
        pass_evaluator_tool = {
            "type": "function",
            "function": {
                "name": "pass_evaluator",
                "description": """Evaluate your AI4C pass optimization implementation.

This tool runs the AI4C evaluation pipeline to check your pass implementation:
1. Pass Matching: Verifies that your pass pattern matches the target computation
2. Correctness: Ensures your optimized kernel produces correct outputs
3. Performance: Measures the speedup achieved by your optimization

The tool evaluates the pass files in the current problem directory (./pass_dir).
The AI4C_PROBLEM_PATH environment variable is automatically set for you.

If any stage fails, the tool reports the failure. If all stages pass, it reports the speedup.
All output is printed to stdout for you to see.

No parameters are required - simply call this tool to evaluate your current pass implementation.""",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        # Always use AI4C tools
        tools = [file_editor, pass_evaluator_tool]

        # Add prompt caching
        if "vertex" not in self.llm_name.lower():
            tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
            breakpoints_remaining = 3
            for message in reversed(messages):
                if message["role"] in ("user", "tool"):
                    if breakpoints_remaining > 0:
                        message["cache_control"] = {"type": "ephemeral"}
                        breakpoints_remaining -= 1
                    else:
                        break

        # Check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None

        messages_ = copy.deepcopy(messages)

        # Query the model
        start_time = time.time()
        retries = 0

        while retries < self.max_retries:
            try:
                response = litellm.completion(
                    model=self.llm_name,
                    tools=tools,
                    messages=messages_,
                    timeout=self.llm_timeout,
                    api_base=self.llm_base_url,
                    temperature=temperature,
                )
                self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e

        exec_time = time.time() - start_time
        return response, exec_time
