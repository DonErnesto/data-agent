from typing import List, Callable, Dict, Any, Union
import os
import pandas as pd
import numpy as np
import datetime
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
DATA_DIR = "data/"

# NEW: flexible pandas tools
## Tools that give the assistant the power to call a wide range of pandas functions, and 
## save the result in a dictionary. 
DATAFRAMES: Dict[str, pd.DataFrame] = {}
ALLOWED_METHODS = {"head", "describe", "mean", "sum", "info", "columns", "min", "max"}


def _json_safe(obj):
    """Recursively convert Pandas/Numpy objects into JSON-safe Python objects."""
    if isinstance(obj, pd.DataFrame):
        # Preserve index if it's not the default RangeIndex (keeps 'count','mean',... for describe)
        if not obj.index.equals(pd.RangeIndex(start=0, stop=len(obj))):
            columns = [_json_safe(c) for c in obj.columns]
            index = [_json_safe(i) for i in obj.index]  # handles MultiIndex/datetimes
            # Convert each cell via _json_safe to handle Timestamps etc.
            data = [[_json_safe(v) for v in row] for row in obj.itertuples(index=False, name=None)]
            return {"columns": columns, "index": index, "data": data}
        else:
            # For "normal" tables, just return as list of records
            return [_json_safe(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "tolist"):  # numpy arrays, etc.
        return obj.tolist()
    return obj



def load_dataframe(path: str, alias: str):
    """Load a dataframe from a path and register it with an alias."""
    if not os.path.isabs(path):
        full_path = os.path.join(DATA_DIR, path)
    else:  #make it accept a full path too for pytest.fixture
        full_path = path
    if not os.path.exists(full_path):
        raise ValueError(f"File not found at path: {full_path}")

    if path.endswith('.csv'):
        df = pd.read_csv(full_path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(full_path)
    else:
        print(f"path: {path}. path endswith csv: {path.endswith('.csv')}")
        raise NotImplementedError("Only .parquet and .csv implemented for reading.")  
    DATAFRAMES[alias] = df
    return f"Dataframe '{alias}' loaded with shape {df.shape}"

def call_dataframe_method(alias: str, method: str, *args, **kwargs):
    """
    Call a whitelisted Pandas method on a DataFrame stored in DATAFRAMES.
    Return JSON-safe result.
    """
    if alias not in DATAFRAMES:
        raise ValueError(f"No dataframe registered with alias '{alias}'")

    df = DATAFRAMES[alias]

    # Whitelist
    allowed = {"head", "describe", "info", "shape", "columns", "mean", "sum"}
    if method not in allowed:
        raise ValueError(f"Method {method} not allowed")

    func = getattr(df, method)

    # Special case: df.info() prints to stdout → capture as string
    if method == "info":
        import io
        buf = io.StringIO()
        df.info(buf=buf)
        return buf.getvalue()

    # Otherwise, call method
    result = func(*args, **kwargs)

    # Convert to JSON-safe
    return _json_safe(result)


def call_column_method(alias: str, column: str, method: str):
    """ Call selected functions on a dataframe column."""
    if alias not in DATAFRAMES:
        raise ValueError(f"No dataframe registered with alias '{alias}'")
    df = DATAFRAMES[alias]
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")

    if method not in {"mean", "sum", "median", "std", "min", "max"}:
        raise ValueError(f"Method {method} not allowed")

    return _json_safe(getattr(df[column], method)())


def merge_dataframes(left: str, right: str, on: str, how: str = "inner", alias: str = None):
    """
    Merge two dataframes by their alias and store result under a new alias.
    """
    df_left = DATAFRAMES[left]
    df_right = DATAFRAMES[right]
    merged = pd.merge(df_left, df_right, on=on, how=how)
    alias = alias or f"{left}_{right}_merged"
    DATAFRAMES[alias] = merged
    return f"Merged dataframe stored as '{alias}' with shape {merged.shape}"


def load_df_from_path(path: str):
    full_path = pathlib.Path(DATA_DIR) / path
    if not os.path.exists(full_path):
        raise ValueError(f"File not found at path: {full_path}")
    if path.endswith('.csv'):
        return pd.read_csv(full_path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(full_path)
    else:
        raise NotImplementedError("Extension not implemented for reading.")

class LoadDataFrameParams(BaseModel):
    """Load a dataframe from a file path and assign an alias."""
    alias: str = Field(..., description="The alias to assign the dataframe to.")
    path: str = Field(..., describe_column="The path to load the dataframe from.")

class CallDataFrameMethodParams(BaseModel):
    """"Call a safe pandas method (e.g., head, describe, mean) on a dataframe by alias. """
    method: str = Field(..., description="The name of the pandas method to call")
    alias: str = Field(..., description="Alias of the dataframe to operate on")
    args: List[Any] = Field(default_factory=list, description="Positional arguments for the method")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the method")

class MergeDataFramesParams(BaseModel):
    """Merge two dataframes by ther alias and store result under a new alias. """
    left: str = Field(..., description="The alias of the left dataframe.")
    right: str = Field(..., description="The alias of the right dataframe.")
    on: str = Field(..., description="On which column merge the two dataframes")
    how: str = Field(..., description="How to merge the two dataframes (per default: 'inner')")
    alias: str = Field(..., description="The alias to store the result under.")

class CallColumnMethodParams(BaseModel):
    """"Call a safe pandas method (e.g., head, describe, mean) on a dataframe by alias. """
    alias: str = Field(..., description="Alias of the dataframe to operate on")
    column: str = Field(..., description="The column on which to apply the method")
    method: str = Field(..., description="The name of the pandas method to call")


# Function definitions. These can be called by the LLM.
def list_files(dir: str = None) -> list:
    """List all files in a directory. If no dir is given, the DATA_PATH is used."""
    if not dir:
        dir = DATA_DIR
    return os.listdir(dir)

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
 

