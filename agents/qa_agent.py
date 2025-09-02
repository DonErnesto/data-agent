import os
from typing import List
from dotenv import load_dotenv
import pathlib
from pydantic import BaseModel, Field
import pandas as pd
from src.agent.environment import Environment
from src.agent.actions import ActionRegistry, Action
from src.agent.actions import list_files, describe_dataframe, show_datatype_of_column, describe_column, compare_similarity_column_joined_on_key
from src.agent.actions import ListFilesParams, DescribeDataframeParams, ShowDatatypeOfColumnParams, DescribeColumnParams, CompareSimilarityColumnJoinedOnKeyParams
from src.agent.agent import Agent, AgentLanguage, AgentFunctionCallingActionLanguage, generate_response
from src.agent.goals import Goal
load_dotenv() # This loads variables from .env into os.environ


goals = [
    Goal(priority=1, name="Gather Information", description="""
    - Find the latest and the previous SBTI target file in data, by listing all parquet files.
    - Get the main characteristics of both files, like data type, and basic statistics.
    - Especially pay attention to changes observed.
     """),
    Goal(priority=1, name="Do a deep-dive in the columns", description="""
    - Based on the information gathered, do a deep dive on the changes in data on column-level
    - Investigate AT LEAST a handful of columns, focusing on those with largest relevance to the dataset.
    - Especially determine the similarity of the columns, and judge whether the changes are significant.
    - Use extra information available to make an expert judgement and report these details in the final summary.
    - Report columns with large changes by explicitly flagging a WARNING, and give background explanations when doing so, 
    for instance by comparing the value counts before and after, and providing some sample values.
    - Note that the SBTI data is joined on the "sbti_id" column, therefore do not compare this column. 
    """),
    Goal(priority=1, 
    name="Terminate", description="Call the terminate function only when you have completed the information gathering "\
        "and the column-wise deep dive. "\
        "Most importantly: give a highly structured and extensive summary of the observations,  "\
        "First in a section listing the "
    )
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

action_registry.register(Action(
    name="compare_similarity_column_joined_on_key",
    function=compare_similarity_column_joined_on_key,
    description="compare the similarity of a previous and current version of a column joined on key. ",
    pydantic_base_model=CompareSimilarityColumnJoinedOnKeyParams,
    terminal=False
))


# Define the environment
environment = Environment()
agent_language = AgentFunctionCallingActionLanguage()

qa_agent = Agent(goals, agent_language, action_registry, generate_response, environment)
user_input = """
You are an AI agent that can perform tasks by using available tools to answer questions about two pandas DataFrames that are loaded in the environment.

Your workflow is:
1. Read the user request carefully.
2. Execute the right pandas tool(s) to obtain the necessary information. You may need multiple tool calls.
3. After gathering the required information, analyze the results directly (do not terminate yet).
4. Formulate a clear, concise, and human-interpretable summary/answer based on the results.
5. ONLY after you have the complete summary/answer, call the "terminate" tool ONCE with the full answer in its 'message' field.

⚠️ Rules:
- Do NOT call "terminate" until you have the full final answer ready.
- Do NOT drop or omit the analysis results. The entire human-facing explanation must be included in the terminate message.
- If you are unsure or need more information, make additional tool calls instead of terminating early.

- Your terminate message must always follow this structure:

  Summary:
  <Plain-language explanation of the findings. Clearly answer the user’s question.>

  Key Results:
  - <Specific value(s) or statistics computed>
  - <Any notable comparisons, anomalies, or patterns>
  - <References to relevant columns or subsets used>

  Notes:
  <Optional section for caveats, assumptions, or next steps if more analysis is needed.>
"""



if __name__ == '__main__':
    # Run the agent with user input
    final_memory = qa_agent.run(user_input)

    # Print the final memory
    print(final_memory.get_memories())