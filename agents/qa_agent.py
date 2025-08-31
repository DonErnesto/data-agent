import os
from typing import List
from dotenv import load_dotenv
import pathlib
from pydantic import BaseModel, Field
import pandas as pd
from src.agent.environment import Environment
from src.agent.actions import ActionRegistry, Action
from src.agent.actions import list_files, list_column_names_of_dataframe, describe_column, describe_dataframe, show_datatype_of_column
from src.agent.actions import ListFilesParams, ListColumnNamesOfDataFrameParams, DescribeColumnParams, DescribeDataframeParams, ShowDatatypeOfColumnParams
from src.agent.agent import Agent, AgentLanguage, AgentFunctionCallingActionLanguage, generate_response
from src.agent.goals import Goal
load_dotenv() # This loads variables from .env into os.environ


goals = [
    Goal(priority=1, name="Gather Information", description="""
    - Find the latest SBTI target file in data, and the previous one, by listing all parquet files.
    - Get the principle characteristics of both files, such as the file size, the number of rows and columns, and main statistics of the columns
    - Summarize the main differences between both files

    """),
    Goal(priority=1, name="Terminate", description="Call the terminate call when you have found the two SBTI files, "\
        "or when there is an indication that the file loading is unsuccessful or you run into other problems.")
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

qa_agent = Agent(goals, agent_language, action_registry, generate_response, environment)
user_input = "Describe the parquet data in the directory, and report the latest and previous files."

if __name__ == '__main__':
    # Run the agent with user input
    final_memory = qa_agent.run(user_input)

    # Print the final memory
    print(final_memory.get_memories())