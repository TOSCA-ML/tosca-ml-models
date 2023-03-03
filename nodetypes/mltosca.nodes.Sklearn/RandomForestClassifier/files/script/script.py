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


def create_model(n_estimators, criterion, max_depth, min_samples_split):
  return RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split)


if __name__ == "__main__":
  output_folder, data_folder, target, criterion, max_depth, min_samples_split, n_estimators = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]

  while not os.path.isfile(data_folder + '/configuration.json'):
    time.sleep(1)
  train_dataframe = read_config_file(data_folder + '/configuration.json')

  model = create_model(int(n_estimators), criterion, int(max_depth), int(min_samples_split))
  model = train_model(model, data_folder + '/' + train_dataframe, target)
  save_model(model, output_folder + '/model.pkl')
