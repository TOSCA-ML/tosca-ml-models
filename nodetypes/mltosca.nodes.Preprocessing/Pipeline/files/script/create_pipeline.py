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


def sort_by_column(df, column_name, ascending):
  return df.sort_values(by=column_name, ascending=('true' == ascending.lower())).reset_index(drop=True)


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
  functions_order = [tuple(param for param in line.split(' | ')) for line in lines]
  pipe_chain = create_pipe_chain(functions_order)
  return pipe_chain


def main():
  filepath = '/home/artjom/mltosca/df.pkl'
  while not os.path.exists(filepath):
    time.sleep(1)
  df = read_dataframe(filepath)
  pipeline = create_pipeline('/home/artjom/mltosca/order.txt')
  df = eval(pipeline)
  df.to_csv('/home/artjom/Downloads/result.csv')


if __name__ == "__main__":
  # main()
  pass
