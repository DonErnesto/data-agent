from typing import List, Callable, Dict, Any, Union
import os
import pandas as pd
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

class TerminateParams(BaseModel):
    """Terminates the conversation. No further actions or interactions are possible after this. Prints the provided message for the user."""
    message: str = Field(..., description="The message to print to the user.")


class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 pydantic_base_model: BaseModel,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.pydantic_base_model = pydantic_base_model
        self.parameters = convert_to_openai_function(
            pydantic_base_model)['parameters']

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)

class ActionRegistry:
    def __init__(self):
        self.actions = {
                "terminate":Action(
                    name="terminate",
                    function=lambda message: f"{message}\nTerminating...",
                    description="Terminates the session and prints the message to the user.",
                    pydantic_base_model=TerminateParams,
                    terminal=True
                )
        }

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> Union[Action, None]:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())



