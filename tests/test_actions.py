import datetime
import json
import os

import pandas as pd
import pytest

from data_agent.agent.actions import (DATAFRAMES, _json_safe,
                                      call_column_method,
                                      call_dataframe_method, list_files,
                                      load_dataframe)

CSV_FILENAME = "sample.csv"

# --- Fixtures ---
@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file with sample data."""
    data = pd.DataFrame(
        {"id": [1, 2, 3], "value": [10, 20, 30], "category": ["A", "B", "A"]}
    )
    file_path = tmp_path / CSV_FILENAME
    data.to_csv(file_path, index=False)
    print(file_path)
    return str(file_path), data  # return both path and original DataFrame


# --- Tests ---
def test_list_files(sample_csv):
    path, _ = sample_csv
    dir = os.path.dirname(path)
    files = list_files(dir)
    assert CSV_FILENAME in set(files)


def test_load_dataframe_registers_alias(sample_csv):
    path, df_expected = sample_csv
    msg = load_dataframe(alias="test_df", path=path)

    assert "test_df" in DATAFRAMES
    df_loaded = DATAFRAMES["test_df"]

    # content matches
    pd.testing.assert_frame_equal(df_loaded, df_expected)

    # check return message
    assert "test_df" in msg
    assert str(df_loaded.shape) in msg


def test_call_dataframe_method_head(sample_csv):
    path, _ = sample_csv
    load_dataframe(alias="test_df", path=path)

    result = call_dataframe_method("test_df", "head", 2)
    # result should be a list of dicts
    assert isinstance(result, list)

    # verify correct number of rows returned
    df_head = pd.DataFrame(result)
    assert df_head.shape[0] == 2


@pytest.mark.parametrize(
    "method,args",
    [
        ("head", (2,)),
        ("describe", ()),
    ],
)
def test_json_serializable_results(sample_csv, method, args):
    path, _ = sample_csv
    load_dataframe(alias="test_df", path=path)
    result = call_dataframe_method("test_df", method, *args)
    try:
        json.dumps(result)
    except TypeError as e:
        pytest.fail(f"Result of {method} not JSON serializable: {e}")


def test_call_dataframe_method_disallowed_method(sample_csv):
    path, _ = sample_csv
    load_dataframe(alias="test_df", path=path)

    with pytest.raises(ValueError) as e:
        call_dataframe_method("test_df", "drop", ["id"])
    assert "not allowed" in str(e.value)


def test_column_mean_non_numeric(sample_csv):
    path, _ = sample_csv
    load_dataframe(alias="test_df", path=path)

    # mean on string column should raise
    with pytest.raises(Exception):
        call_dataframe_method("test_df", "mean", column="category")


@pytest.mark.parametrize(
    "method,args", [("mean", ()), ("max", ()), ("min", ()), ("std", ())]
)
def test_json_serializable_column_results(sample_csv, method, args):
    path, _ = sample_csv
    load_dataframe(alias="test_df", path=path)
    result = call_column_method("test_df", "value", method, *args)
    try:
        json.dumps(result)
    except TypeError as e:
        pytest.fail(f"Result of {method} not JSON serializable: {e}")


def test_json_safe_dataframe_with_timestamps():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "when": pd.to_datetime(["2020-01-01 12:00:00", "2020-01-02 15:30:00"]),
            "date_only": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)],
        }
    )

    # Use describe to stress-test stats including datetime columns
    desc = df.describe(include="all")
    safe = _json_safe(desc)
    json_str = json.dumps(safe)  # should not raise
    assert "2020-01-01" in json_str or "2020-01-02" in json_str
    assert isinstance(safe, (dict, list))  # structure is JSON safe


def test_json_safe_correct():
    df = pd.DataFrame(
        {"id": [1, 2, 3], "value": [10, 20, 30], "category": ["A", "B", "A"]}
    )
    safe = _json_safe(df)
    df_reconstructed = pd.DataFrame.from_dict(safe)
    pd.testing.assert_frame_equal(df, df_reconstructed)
    print(safe)


def test_json_safe_describe_correct():
    data = pd.DataFrame(
        {"id": [1, 2, 3], "value": [10, 20, 30], "category": ["A", "B", "A"]}
    )
    df = data.describe()
    print(df)
    safe = _json_safe(df)

    df_reconstructed = pd.DataFrame.from_dict(data=safe["data"], orient="columns")
    df_reconstructed.columns = safe["columns"]
    df_reconstructed.index = safe["index"]
    print(f"df: {df}")
    print(f"df reconstructed: {df_reconstructed}")
    print(f"safe: {safe}")
    pd.testing.assert_frame_equal(df, df_reconstructed)


def test_call_dataframe_method_attribute_and_method(sample_csv):
    path, _ = sample_csv
    load_dataframe(alias="test_df", path=path)

    # Attribute
    cols = call_dataframe_method("test_df", "columns")
    assert isinstance(cols, list)  # JSON-safe

    # Method
    head_rows = call_dataframe_method("test_df", "head", 2)
    assert isinstance(head_rows, list)
    assert len(head_rows) == 2


if __name__ == "__main__":

    test_json_safe_dataframe_with_timestamps()
    test_json_safe_correct()
    test_json_safe_describe_correct()

    data = pd.DataFrame(
        {"id": [1, 2, 3], "value": [10, 20, 30], "category": ["A", "B", "A"]}
    )
    DATAFRAMES["test_df"] = data
    cols = call_dataframe_method("test_df", "columns")
