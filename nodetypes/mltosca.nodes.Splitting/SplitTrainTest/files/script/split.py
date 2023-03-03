import os
import sys
import time
import json
import pandas as pd
from pandas import DataFrame


def setup_config_file(project_location, train_dataframe='train.pkl', test_dataframe='test.pkl'):
  with open(project_location + '/configuration.json', 'w') as file:
    json.dump({'filenames': [train_dataframe, test_dataframe]}, file, indent=4)


def read_config_file(config_path) -> str:
  with open(config_path, 'r') as f:
    config = json.load(f)
  return config['filenames'][0]


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
  setup_config_file(output_folder)

  while not os.path.isfile(previous_output_folder + '/configuration.json'):
    time.sleep(1)

  dataframe_file = read_config_file(previous_output_folder + '/configuration.json')

  while not os.path.isfile(previous_output_folder + '/' +dataframe_file):
    time.sleep(1)

  df = read_dataframe(previous_output_folder + '/' +dataframe_file)
  train, test = train_test_split(df, test_size)

  train.to_pickle(output_folder + '/processing.pkl')
  os.rename(output_folder + '/processing.pkl', output_folder + '/train.pkl')

  test.to_pickle(output_folder + '/processing.pkl')
  os.rename(output_folder + '/processing.pkl', output_folder + '/test.pkl')
