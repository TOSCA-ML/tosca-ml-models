import os
import sys
import time
import json
import pickle
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression


def save_model(model, file_path):
  with open(file_path, 'wb') as file:
    pickle.dump(model, file)


def read_dataframe(path: str) -> DataFrame:
  return pd.read_pickle(path)


def transform_boolean(variable: str):
  return variable.lower() == "true"


def setup_config_file(project_location, dataframe_name='model.pkl'):
  with open(project_location + '/configuration.json', 'w') as file:
    json.dump({'filenames': [project_location + '/' + dataframe_name]}, file, indent=4)


def read_config_file(config_path) -> str:
  with open(config_path, 'r') as f:
    config = json.load(f)
  return config['filenames'][0]


def train_model(model, train_data, target):
  while not os.path.isfile(train_data):
    time.sleep(1)
  df = read_dataframe(train_data)
  X = df.drop(target, axis=1)
  y = df[target]
  model.fit(X, y)
  return model


def create_model(fit_intercept, copy_X, positive):
  return LinearRegression(fit_intercept=transform_boolean(fit_intercept),
                          copy_X=transform_boolean(copy_X),
                          positive=transform_boolean(positive))


if __name__ == "__main__":
  output_folder, data_folder, target, fit_intercept, copy_X, positive = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
  setup_config_file(output_folder)

  while not os.path.isfile(data_folder + '/configuration.json'):
    time.sleep(1)
  train_dataframe = read_config_file(data_folder + '/configuration.json')

  model = create_model(fit_intercept, copy_X, positive)
  model = train_model(model, train_dataframe, target)
  save_model(model, output_folder + '/model.pkl')
