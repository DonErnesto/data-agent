from typing import List
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

# Define your Pydantic models
class ListColumnNamesOfDataFrameTool(BaseModel):
    """List column names of a pandas DataFrame."""
    pass

class ShowDatatypeOfColumnTool(BaseModel):
    """Show the datatype of a column in a pandas DataFrame."""
    column_name: str = Field(..., description="The name of the column to show the datatype of.")

class DescribeColumnTool(BaseModel):
    """Describe the contents of a column in a pandas DataFrame."""
    column_name: str = Field(..., description="The name of the column to describe.")

class TerminateTool(BaseModel):
    """Terminates the conversation. No further actions or interactions are possible after this. Prints the provided message for the user."""
    message: str = Field(..., description="The message to print to the user.")


# Create a list of your Pydantic models and their corresponding desired function names
tool_models_and_names = [
    (ListColumnNamesOfDataFrameTool, "list_column_names_of_dataframe"),
    (ShowDatatypeOfColumnTool, "show_datatype_of_column"),
    (DescribeColumnTool, "describe_column"),
    (TerminateTool, "terminate"),
]

# Convert each Pydantic model to the OpenAI function format, specifying the name
tools = []
for tool_model, tool_name in tool_models_and_names:
    openai_function_def = convert_to_openai_function(tool_model)
    # Manually set the name in the generated function definition
    openai_function_def['name'] = tool_name
    # Wrap in the full tool structure as needed by OpenAI (as seen in your output)
    tools.append({
        "type": "function",
        "function": openai_function_def
    })

