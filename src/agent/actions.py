from typing import List, Callable, Dict, Any, Union
import os
import pandas as pd
import pathlib
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

class TerminateParams(BaseModel):
    """Terminates the conversation and provides the final analysis to the user. """
    message: str = Field(..., description="The final analysis to provide to the user.")


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
                    function=lambda message: f"{message} \n Terminating...",
                    description="Terminates the session and prints the final to the user.",
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
    if path.endswith('.csv'):
        return pd.read_csv(full_path)
    if path.endswith('.parquet'):
        return pd.read_parquet(full_path)
    else:
        raise NotImplementedError("Extension not implemented for reading.")

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
    return df.describe(include='all').T[["count", "unique", "freq", "mean", "std"]].to_string()

def show_datatype_of_column(path: str, column_name: str) -> str:
    """Show the datatype of a column in a pandas DataFrame."""
    df = load_df_from_path(path)
    return str(df[column_name].dtype)

def describe_column(path: str, column_name: str) -> str:
    """Describe the contents of a column in a pandas DataFrame."""
    df = load_df_from_path(path)
    description_column =  df[column_name].describe().to_string()
    normalized_value_counts = df[column_name].value_counts(normalize=True, dropna=False)
    normalized_perc = ((normalized_value_counts * 100).map('{:.3f}%'.format).to_string())
    return f"Description of column: {description_column} \n \n Normalized value counts: {normalized_perc}"

def translate_pd_to_human(message) -> None:
    """ Translate the pandas results into a human-readable text.
    This will terminate the loop.
    """
    print(f"The pandas results can be described as follows: {message}")

# Functions that compare compare columns after joining on a primary key
def outer_join_on_key(df_1_path: str, df_2_path: str, join_key='sbti_id'):
    df_1 = load_df_from_path(df_1_path)
    df_2 = load_df_from_path(df_2_path)
    return df_1.merge(df_2, how='outer', on='sbti_id', suffixes=('_prev', '_curr'), indicator=True)

def compare_similarity_column_joined_on_key(path_df_prev: str, path_df_curr: str, column_name: str) -> str:
    joined_df = outer_join_on_key(path_df_prev, path_df_curr, join_key='sbti_id')
    normalized_counts = joined_df["_merge"].value_counts(normalize=True)

    # Format as percentages
    percentage_merge_stats = (normalized_counts * 100).map('{:.3f}%'.format).to_string()
    both_joined_idx = joined_df["_merge"] == "both"
    number_joined = both_joined_idx.sum()

    both_joined_frac_diff = (joined_df.loc[both_joined_idx, f"{column_name}_prev"] !=
                    joined_df.loc[both_joined_idx, f"{column_name}_curr"]).sum()/number_joined
    both_joined_percent_diff = f"{both_joined_frac_diff:.3%}"

    merge_str = f"The percentages of merges (both, only old, only new) are: \n {percentage_merge_stats}\n"
    identical_str = f"The percentage of values that could be merged that are unequal is {both_joined_percent_diff}"
    return (f"Analyzed similarity of column {column_name}: {merge_str}. {identical_str}.")


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

class CompareSimilarityColumnJoinedOnKeyParams(BaseModel):
    """ Get join- and similarity metrics for a column after joining on a primary key"""
    path_df_prev: str = Field(..., description="The path to the previous data.")
    path_df_curr: str = Field(..., description="The path to the current data")
    column_name: str = Field(..., description="The name of the column to compare.")



