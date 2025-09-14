from dotenv import load_dotenv

from data_agent.agent.actions import (
    Action,
    ActionRegistry,
    CallColumnMethodParams,
    CallDataFrameMethodParams,
    ListFilesParams,
    LoadDataFrameParams,
    MergeDataFramesParams,
    call_column_method,
    call_dataframe_method,
    list_files,
    load_dataframe,
    merge_dataframes,
)
from data_agent.agent.agent import (
    Agent,
    AgentFunctionCallingActionLanguage,
    generate_response,
)
from data_agent.agent.environment import Environment
from data_agent.agent.goals import Goal

load_dotenv()  # This loads variables from .env into os.environ

goals = [
    Goal(
        priority=1,
        name="Explore the data and find columns that exhibit large changes",
        description="""
    - Gather high-level statistics like size, number of columns, column names, of the dataframes.
    - Use functions like 'describe' on the dataframes to detect columns that may have a large change
     from previous to current values
    - Use functions like 'value_counts' on selected columns to understand the nature of the changes.
    """,
    ),
    Goal(
        priority=1,
        name="Do detailed comparisons between columns that show indication of large changes",
        description="""
    - By loading the data frame and assigning it an alias, you can access it.
    - The alias is especially useful for merged data frames.
    - Merge data frames on columns that look like identifiers, for instance with "id" in the name.
    - After merging, do a groupby on categorical columns to understand the nature of the changes.
    - For numerical columns, do a correlation to detect large changes.
    - Larger changes (exceeding a few percent) need flagging by a WARNING.
    - Standard pandas functions for dataframes at your disposal:
        "describe", "mean", "max", "min", "merge", "groupby".
    - Continue searching for changes until you have a complete analysis of all changes.
    """,
    ),
    Goal(
        priority=1,
        name="Terminate and report",
        description="""
    - Call the terminate function only when you have gathered all information,
    including detailed column-wise comparisons
    - Give a highly structured and extensive summary of the observations
    - Give explicit WARNING when changes larger than a few percent are found that may indicate
    large shifts in the data.
        """,
    ),
]

# Create and populate the action registry
action_registry = ActionRegistry()

action_registry.register(
    Action(
        name="list_files",
        function=list_files,
        description="List all files in the current directory",
        pydantic_base_model=ListFilesParams,
        terminal=False,
    )
)

action_registry.register(
    Action(
        name="load_dataframe",
        function=load_dataframe,
        description="Load a dataframe and store under an alias.",
        pydantic_base_model=LoadDataFrameParams,
        terminal=False,
    )
)

action_registry.register(
    Action(
        name="call_dataframe_method",
        function=call_dataframe_method,
        description="Call a safe method on a dataframe registered under an alias." "",
        pydantic_base_model=CallDataFrameMethodParams,
        terminal=False,
    )
)

action_registry.register(
    Action(
        name="call_column_method",
        function=call_column_method,
        description="Apply a method such as mean, min, max, or value_counts to a single column." "",
        pydantic_base_model=CallColumnMethodParams,
        terminal=False,
    )
)

action_registry.register(
    Action(
        name="merge_dataframes",
        function=merge_dataframes,
        description="Merge two dataframes by their alias and store result under a new alias.",
        pydantic_base_model=MergeDataFramesParams,
        terminal=False,
    )
)

# Define the environment
environment = Environment()
agent_language = AgentFunctionCallingActionLanguage()

data_scientist = Agent(goals, agent_language, action_registry, generate_response, environment)

user_input = """
You are an AI agent that can perform tasks by using available tools to answer questions about files
present in the environment.

Your workflow is:
1. Read the user request carefully.
2. Execute the right pandas tool(s) to obtain the necessary information. Multiple calls are needed.
3. After gathering the required information, analyze the results directly (do not terminate yet).
4. Do extensive column-level comparisons and analyze these as well (do not terminate yet).
4. Formulate a clear, concise, and human-interpretable summary/answer based on the results.
5. ONLY after you have the complete summary/answer, call the "terminate" tool ONCE with the full
   answer in its 'message' field.

⚠️ Rules:
- Do NOT call "terminate" until you have the full final answer ready.
- Do NOT drop or omit the analysis results. The entire human-facing explanation must be included
  in the terminate message.
- If you are unsure or need more information, make additional tool calls instead of terminating
  early.

- Your terminate message must always follow this structure:

  Summary:
  <Plain-language explanation of the findings. Clearly answer the user’s question.>

  Key Results:
  - <Specific value(s) or statistics computed>
  - <Any notable comparisons, anomalies, or patterns>
  - <References to relevant columns or subsets used>

  WARNINGS:
  - <List all large changes that have been observed and that may warrant closer inspection>

  Notes:
  <Optional section for caveats, assumptions, or next steps if more analysis is needed.>
"""
