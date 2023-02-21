import sys
import pandas as pd
from pandas import DataFrame


def read_dataframe(path: str) -> DataFrame:
    return pd.read_pickle(path)


def save_dataframe(df: DataFrame, path: str):
    df.to_pickle(path)


def keep_columns(df: DataFrame, columns: str) -> DataFrame:
    columns = [column.strip() for column in columns.split(',')]
    return df[columns]


if __name__ == "__main__":
    filePath, columns = sys.argv[1:]
    df = read_dataframe(filePath)
    df = keep_columns(df, columns)
    save_dataframe(df, filePath)
