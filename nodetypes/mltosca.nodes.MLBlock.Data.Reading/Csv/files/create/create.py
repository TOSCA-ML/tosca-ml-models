import sys
import pandas as pd
from pandas import DataFrame


def read_csv(filepath) -> DataFrame:
  return pd.read_csv(filepath)


def save_dataframe(df: DataFrame):
  df.to_pickle('df.pkl')


if __name__ == "__main__":
  csvPath = sys.argv[1]
  df = read_csv(csvPath)
  save_dataframe(df)

