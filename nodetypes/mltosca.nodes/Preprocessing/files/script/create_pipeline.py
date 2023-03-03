import sys
import json
import time
import os.path
import pandas as pd
from pandas import DataFrame


def remove_columns(df: DataFrame, columns: str) -> DataFrame:
  columns = [column.strip() for column in columns.split(',')]
  return df.drop(columns=columns, axis=1)


def keep_columns(df: DataFrame, columns: str) -> DataFrame:
  columns = [column.strip() for column in columns.split(',')]
  return df[columns]


def sort_by_column(df, column_name, order):
  return df.sort_values(by=column_name, ascending=('ascending' == order.lower())).reset_index(drop=True)


def rename_column(df: DataFrame, old_col_name: str, new_col_name: str) -> DataFrame:
  return df.rename(columns={old_col_name: new_col_name})


def remove_nan_rows(df: DataFrame) -> DataFrame:
  return df.dropna().reset_index(drop=True)


def shuffle_rows(df: DataFrame) -> DataFrame:
  return df.sample(frac=1)


def replace_categorical_values(df: DataFrame, column: str) -> DataFrame:
  unique_values = df[column].unique()
  value_map = {value: i for i, value in enumerate(unique_values)}
  df[column] = df[column].map(value_map)
  return df


def replace_nan_values(df: DataFrame, column: str, method: str, value: str = None) -> DataFrame:
  def assign_type(df: DataFrame, column: str, value: str):
    dtype = df[column].dtype
    if pd.api.types.is_numeric_dtype(dtype):
      return int(value)
    elif pd.api.types.is_float_dtype(dtype):
      return float(value)
    return value

  if method == 'mean':
    df[column] = df[column].fillna(df[column].mean())
  elif method == 'median':
    df[column] = df[column].fillna(df[column].median())
  elif method == 'mode':
    df[column] = df[column].fillna(df[column].mode().iloc[0])
  elif method == 'value' and value is not None:
    value = assign_type(df, column, value)
    df[column] = df[column].fillna(value)
  return df


def read_dataframe(path: str) -> DataFrame:
  return pd.read_pickle(path)


def create_pipe_chain(pipeline: list) -> str:
  pipe_chain = 'df'
  for function in pipeline:
    func_str = f'.pipe({function[0]}'
    if len(function) > 1:
      for param in function[1:]:
        func_str += f", '{param}'"
    func_str += ')'
    pipe_chain += func_str
  return pipe_chain


def create_pipeline(path: str) -> str:
  with open(path) as file:
    lines = [line.rstrip() for line in file]
  functions_order = [tuple(param for param in line.split('-|-')) for line in lines]
  pipe_chain = create_pipe_chain(functions_order)
  return pipe_chain


def count_lines(filename: str) -> int:
  with open(filename, 'r') as f:
    lines = f.readlines()
    return len(lines)


def read_config_file(config_path) -> str:
  with open(config_path, 'r') as f:
    config = json.load(f)
  return config['filenames'][0]


def setup_config_file(project_location, dataframe_name='df.pkl'):
  with open(project_location + '/configuration.json', 'w') as file:
    json.dump({'filenames': [dataframe_name]}, file, indent=4)


def main():
  output_folder, previous_output_folder = sys.argv[1], sys.argv[2]
  setup_config_file(output_folder)
  functions_count = count_lines(output_folder + '/count.txt')

  while True:
    if os.path.isfile(output_folder + '/order.txt'):
      with open(output_folder + '/order.txt', 'r') as f:
        lines = f.readlines()
        if len(lines) == functions_count:
          break

  while not os.path.isfile(previous_output_folder + '/configuration.json'):
    time.sleep(1)

  dataframe_file = read_config_file(previous_output_folder + '/configuration.json')

  while not os.path.isfile(previous_output_folder + '/' + dataframe_file):
    time.sleep(1)
  df = read_dataframe(previous_output_folder + '/' + dataframe_file)

  pipeline = create_pipeline(output_folder + '/order.txt')
  df = eval(pipeline)

  df.to_pickle(output_folder + '/processing.pkl')
  os.rename(output_folder + '/processing.pkl', output_folder + '/df.pkl')


if __name__ == "__main__":
  main()
