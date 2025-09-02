import os
from typing import List
from dotenv import load_dotenv
import pathlib
from pydantic import BaseModel, Field
import pandas as pd
from ..agent.environment import Environment
from ..agent.actions import ActionRegistry, Action
from ..agent.actions import list_files, list_column_names_of_dataframe, describe_column, describe_dataframe, show_datatype_of_column
from ..agent.actions import ListFilesParams, ListColumnNamesOfDataFrameParams, DescribeColumnParams, DescribeDataframeParams, ShowDatatypeOfColumnParams
from ..agent.agent import Agent, AgentLanguage, AgentFunctionCallingActionLanguage, generate_response
from ..agent.goals import Goal
load_dotenv() # This loads variables from .env into os.environ

goals = [
    Goal(priority=1, name="Gather Information", description="Giving a summary of all data present in the data directory, "\
    "by listing all files in the directory, and describing the dataframes. "),
    Goal(priority=1, name="Terminate", description="Call the terminate call when you have descriptions of all dataframes, "\
        "or when there is an indication that the file loading is unsuccessful or you run into other problems.")
]

#use for testing: terminates immediately.
terminate_goals = [
    Goal(priority=1, name="Terminate", description="Call the terminate call directly, and close off with a joke."),
]

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

describe_agent = Agent(goals, agent_language, action_registry, generate_response, environment)
terminate_agent = Agent(terminate_goals, agent_language, action_registry, generate_response, environment)
if __name__ == '__main__':
    # Run the agent with user input
    user_input = "Describe the data in this directory."
    final_memory = describe_agent.run(user_input)

    # Print the final memory
    print(final_memory.get_memories())