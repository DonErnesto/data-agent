import os
from typing import List
from pydantic import BaseModel, Field
import pandas as pd
from src.agent.environment import Environment
from src.agent.actions import ActionRegistry, Action
from src.agent.agent import Agent, AgentLanguage, AgentFunctionCallingActionLanguage, generate_response
from src.agent.goals import Goal
from dotenv import load_dotenv
load_dotenv() # This loads variables from .env into os.environ

# Define a simple file management goal


goals = [
    Goal(priority=1, name="Gather Information", description="Giving a summary of all data present in the data directory, "\
    "by listing all files in the directory, and describing the dataframes. "),
    Goal(priority=0, name="Terminate", description="Call the terminate call when you have descriptions of all dataframes, "\
        "or when there is an indication that the file loading is unsuccessful or you run into other problems.")
]

# Helper functions (may be migrated to utils at some point)
def load_df_from_path(path: str):
    if not os.path.exists(path):
        raise ValueError(f"File not found at path: {path}")
    return pd.read_csv(path)

# Function definitions. These can be called by the LLM.
def list_files() -> list:
    """List all files in the data directory."""
    return os.listdir('data/')

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




# Create and populate the action registry
action_registry = ActionRegistry()

action_registry.register(Action(
    name="list_files",
    function=list_files,
    description="List all files in the current directory",
    pydantic_base_model=ListFilesParams,
    terminal=False
))

action_registry.register(Action(
    name="list_column_names_of_dataframe",
    function=list_column_names_of_dataframe,
    description="List the columns of the dataframe",
    pydantic_base_model=ListColumnNamesOfDataFrameParams,
    terminal=False
))

action_registry.register(Action(
    name="describe_dataframe",
    function=describe_dataframe,
    description="Describe the dataframe",
    pydantic_base_model=DescribeDataframeParams,
    terminal=False
))

action_registry.register(Action(
    name="show_datatype_of_column",
    function=show_datatype_of_column,
    description="Show the datatypes of a particular column. ",
    pydantic_base_model=ShowDatatypeOfColumnParams,
    terminal=False
))

action_registry.register(Action(
    name="describe_column",
    function=describe_column,
    description="Describe a particular column in the dataframe",
    pydantic_base_model=DescribeColumnParams,
    terminal=False
))


# Define the environment
environment = Environment()
agent_language = AgentFunctionCallingActionLanguage()

if __name__ == '__main__':


    # Create an agent instance
    agent = Agent(goals, agent_language, action_registry, generate_response, environment)

    # Run the agent with user input
    user_input = "Describe the data in this directory."
    final_memory = agent.run(user_input)

    # Print the final memory
    print(final_memory.get_memories())