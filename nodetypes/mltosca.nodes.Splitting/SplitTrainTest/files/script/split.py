import os
import sys
import time
import pandas as pd
from pandas import DataFrame


def read_dataframe(path: str) -> DataFrame:
  return pd.read_pickle(path)


def train_test_split(df: DataFrame, test_size: float):
  test = int(len(df) * test_size)
  train = len(df) - test
  test_indices = range(train, len(df))
  train_indices = range(train)
  return df.iloc[list(train_indices)], df.iloc[list(test_indices)]


if __name__ == "__main__":
  output_folder, previous_output_folder, test_size = sys.argv[1], sys.argv[2], float(sys.argv[3])

  file_location = previous_output_folder + '/df.pkl'

  while not os.path.isfile(file_location):
    time.sleep(1)
  df = read_dataframe(file_location)
  train, test = train_test_split(df, test_size)

  train.to_pickle(output_folder + '/processing.pkl')
  os.rename(output_folder + '/processing.pkl', output_folder + '/train.pkl')

  test.to_pickle(output_folder + '/processing.pkl')
  os.rename(output_folder + '/processing.pkl', output_folder + '/test.pkl')
