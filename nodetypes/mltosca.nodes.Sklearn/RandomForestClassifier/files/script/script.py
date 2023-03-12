import os
import sys
import time
import json
import pickle
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier


def save_model(model, file_path):
  with open(file_path, 'wb') as file:
    pickle.dump(model, file)


def read_dataframe(path: str) -> DataFrame:
  return pd.read_pickle(path)


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


def create_model(parameters):
  #   criterion, min_samples_split, max_depth, n_estimators = parameters.split('#_#')
  #   return RandomForestClassifier(n_estimators=int(n_estimators),
  #                                 criterion=criterion,
  #                                 max_depth=int(max_depth),
  #                                 min_samples_split=int(min_samples_split))
  return RandomForestClassifier()

if __name__ == "__main__":
  output_folder, data_folder, target, parameters = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

  while not os.path.isfile(data_folder + '/configuration.json'):
    time.sleep(1)
  train_dataframe = read_config_file(data_folder + '/configuration.json')

  model = create_model(parameters)
  model = train_model(model, train_dataframe, target)
  save_model(model, output_folder + '/model.pkl')
