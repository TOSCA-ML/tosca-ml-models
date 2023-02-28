import os
import sys
import pandas as pd
from pandas import DataFrame


def read_csv(filepath, delimiter=',', encoding='UTF-8') -> DataFrame:
    return pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)


def save_dataframe(df: DataFrame, project_location: str):
    df.to_pickle(project_location + '/processing.pkl')
    os.rename(project_location + '/processing.pkl', project_location + '/df.pkl')


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Not enough args")
    project_location, filepath, delimiter, encoding = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    df = read_csv(filepath, delimiter, encoding)
    save_dataframe(df, project_location)
