from typing import List, Callable, Dict, Any, Union
import os
import pandas as pd
import pathlib
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


## Actions
data_dir = "data/"

def load_df_from_path(path: str):
    full_path = pathlib.Path(data_dir) / path
    if not os.path.exists(full_path):
        raise ValueError(f"File not found at path: {full_path}")
    return pd.read_csv(full_path)

# Function definitions. These can be called by the LLM.
def list_files() -> list:
    """List all files in the data directory."""
    return os.listdir(data_dir)

def list_column_names_of_dataframe(path: str) -> List[str]:
    """List column names of a pandas DataFrame."""
    df = load_df_from_path(path)
    return list(df.columns)

def describe_dataframe(path: str) -> str:
    """Describe the contents of a pandas DataFrame."""
    df = load_df_from_path(path)
    return df.describe().to_string()

def show_datatype_of_column(path: str, column_name: str) -> str:
    """Show the datatype of a column in a pandas DataFrame."""
    df = load_df_from_path(path)
    return str(df[column_name].dtype)

def describe_column(path: str, column_name: str) -> str:
    """Describe the contents of a column in a pandas DataFrame."""
    df = load_df_from_path(path)
    return df[column_name].describe().to_string()

def translate_pd_to_human(message) -> None:
    """ Translate the pandas results into a human-readable text.
    This will terminate the loop.
    """
    print(f"The pandas results can be described as follows: {message}")


# Parameter definitions in Pydantic.
class ListFilesParams(BaseModel):
    """List files in the data directory"""
    pass

class ListColumnNamesOfDataFrameParams(BaseModel):
    """List column names of a pandas DataFrame."""
    path: str = Field(..., description="The path to the dataframe to read from")

class DescribeDataframeParams(BaseModel):
    """Describe the contents of the dataframe"""
    path: str = Field(..., description="The path to the dataframe to read from")

class ShowDatatypeOfColumnParams(BaseModel):
    """Show the datatype of a column in a pandas DataFrame."""
    path: str = Field(..., description="The path to the dataframe to read from")
    column_name: str = Field(..., description="The name of the column to show the datatype of.")

class DescribeColumnParams(BaseModel):
    """Describe the contents of a column in a pandas DataFrame."""
    path: str = Field(..., description="The path to the dataframe to read from")
    column_name: str = Field(..., description="The name of the column to describe.")

