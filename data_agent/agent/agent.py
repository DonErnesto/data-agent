import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from litellm import completion

from ..utils.logger import CustomLogger
from .actions import Action, ActionRegistry
from .environment import Environment
from .goals import Goal
from .memory import Memory

logger = CustomLogger(console_level="INFO", file_level="DEBUG")


@dataclass
class Prompt:
    messages: List[Dict] = field(default_factory=list)
    tools: List[Dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Fixing mutable default issue


def generate_response(prompt: Prompt) -> str:
    """Call LLM to get response"""
    messages = prompt.messages
    tools = prompt.tools

    result = None

    if not tools:
        response = completion(
            model="openai/gpt-4o", messages=messages, temperature=0.1, max_tokens=1024
        )
        result = response.choices[0].message.content
    else:
        response = completion(
            model="openai/gpt-4-turbo-2024-04-09",
            messages=messages,
            temperature=0.1,
            tools=tools,
            max_tokens=1024,
        )

        # --- Save the raw response object to disk ---
        # Use a timestamp for the filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        response_filename = f"tmp/raw_llm_response_{timestamp_str}.pkl"
        try:
            with open(response_filename, "wb") as f:
                pickle.dump(response, f)
            logger.info(f"Saved raw LLM response to {response_filename}")
            # To load this file later in another cell:
            # import pickle
            # with open('raw_llm_response_YYYYMMDD_HHMMSS.pkl', 'rb') as f:
            #     loaded_response = pickle.load(f)
            # print(loaded_response) # You can then inspect the loaded_response object
        except Exception as save_e:
            logger.error(
                f"Error saving raw LLM response to {response_filename}: {save_e}",
                exc_info=True,
            )
        # --- End of saving ---

        if response.choices[0].message.tool_calls:
            tool = response.choices[0].message.tool_calls[0]
            result = {
                "tool": tool.function.name,
                "args": json.loads(tool.function.arguments),
            }
            result = json.dumps(result)

        else:
            result = response.choices[0].message.content
            logger.debug(
                f"DEBUG generate_response. NOT TOOL CALL. response: {response}"
            )
            logger.debug(f"DEBUG generate_response. NOT TOOL CALL. result: {result}")

    return result


class AgentLanguage:
    def __init__(self):
        pass

    def construct_prompt(
        self,
        actions: List[Action],
        environment: Environment,
        goals: List[Goal],
        memory: Memory,
    ) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")

    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


class AgentFunctionCallingActionLanguage(AgentLanguage):

    def __init__(self):
        super().__init__()

    def format_goals(self, goals: List[Goal]) -> List:
        # Map all goals to a single string that concatenates their description
        # and combine into a single message of type system
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join(
            [f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals]
        )
        return [{"role": "system", "content": goal_instructions}]

    def format_memory(self, memory: Memory) -> List:
        """Generate response from language model"""
        # Map all environment results to a role:user messages
        # Map all assistant messages to a role:assistant messages
        # Map all user messages to a role:user messages
        items = memory.get_memories()
        mapped_items = []
        for item in items:

            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)

            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item["type"] == "environment":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items

    def format_actions(self, actions: List[Action]) -> [List, List]:
        """Generate response from language model"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    # Include up to 1024 characters of the description
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            }
            for action in actions
        ]

        return tools

    def construct_prompt(
        self,
        actions: List[Action],
        environment: Environment,
        goals: List[Goal],
        memory: Memory,
    ) -> Prompt:

        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)

        tools = self.format_actions(actions)

        return Prompt(messages=prompt, tools=tools)

    def adapt_prompt_after_parsing_error(
        self,
        prompt: Prompt,
        response: str,
        traceback: str,
        error: Any,
        retries_left: int,
    ) -> Prompt:

        return prompt

    def parse_response(self, response: str) -> dict:
        """Parse LLM response into structured format by extracting the ```json block"""
        try:
            return json.loads(response)
        except Exception:
            logger.debug(f"DEBUG parse_response === response: {response}.")
            return {
                "tool": "escalate_incorrect_response",
                "args": {"message": response},
            }


class Agent:
    def __init__(
        self,
        goals: List[Goal],
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        generate_response: Callable[[Prompt], str],
        environment: Environment,
    ):
        """
        Initialize an agent with its core GAME components
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment

    def construct_prompt(
        self, goals: List[Goal], memory: Memory, actions: ActionRegistry
    ) -> Prompt:
        """Build prompt with memory context"""
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory,
        )

    def get_action(self, response):
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        action_def, _ = self.get_action(response)
        try:
            return action_def.terminal
        except AttributeError:
            return True

    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        """
        Update memory with the agent's decision and the environment's response.
        """
        new_memories = [
            {"type": "assistant", "content": response},
            {"type": "user", "content": json.dumps(result)},
        ]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
        response = self.generate_response(full_prompt)
        logger.debug(f"===DEBUG response ==== : {response} === DEBUG response END===")
        return response

    def run(self, user_input: str, memory=None, max_iterations: int = 50) -> Memory:
        """
        Execute the GAME loop for this agent with a maximum iteration limit.
        """
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        for i in range(max_iterations):
            logger.info(f"--- Agent Iteration {i+1} ---")
            # print(f"Iteration {i}")
            # Construct a prompt that includes the Goals, Actions, and the current Memory
            prompt = self.construct_prompt(self.goals, memory, self.actions)

            logger.info("Agent thinking...")
            # Generate a response from the agent
            response = self.prompt_llm_for_action(prompt)

            logger.debug(f"Agent Decision: {response}")
            should_terminate = self.should_terminate(response)
            # Determine which action the agent wants to execute
            action, invocation = self.get_action(response)
            logger.debug(f"DEBUG main loop. action = {action}")
            logger.debug(f"DEBUG main loop. invocation = {invocation}")

            if invocation["tool"] == "escalate_incorrect_response":
                result = f"""The following response could not be processed: {response}.
                        Please ensure you provide only one, correct action. """
                self.update_memory(memory, response, result)
                logger.warning(f"Action result: {result}")
                continue

            logger.debug(response)

            if not should_terminate:
                try:
                    tool = invocation["tool"]
                    args = invocation["args"]
                    logger.info(f"Agent decision: use tool {tool} with args {args}")
                except TypeError:
                    logger.warning("Couldn't parse the response!")

            # Execute the action in the environment
            result = self.environment.execute_action(action, invocation["args"])
            logger.debug(f"Action Result: {result}")

            # Update the agent's memory with information about what happened
            self.update_memory(memory, response, result)

            # Check if the agent has decided to terminate
            if should_terminate:
                logger.info("Agent decided to terminate.")
                print(result["result"])
                break
            elif i == max_iterations - 1:
                logger.warning(f"Max iterations ({max_iterations}) reached.")

        return memory
