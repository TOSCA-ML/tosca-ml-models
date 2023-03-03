import os
import sys
import time
import json
import pickle
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score


def read_config_file(config_path: str) -> str:
  with open(config_path, 'r') as f:
    config = json.load(f)
  return config['filenames']


def wait_for_model(model_folder: str):
  while not os.path.isfile(model_folder + '/model.pkl'):
    time.sleep(1)
  model_name = 'model.pkl'
  with open(model_folder + '/' + model_name, 'rb') as file:
    return pickle.load(file)


def wait_for_data(data_folder: str) -> DataFrame:
  while not os.path.isfile(data_folder + '/configuration.json'):
    time.sleep(1)
  data = read_config_file(data_folder + '/configuration.json')
  file_name = 'test.pkl' if 'test.pkl' in data else data[0]
  return pd.read_pickle(data_folder + '/' + file_name)


def read_evaluation_functions(output_folder: str):
  while not os.path.isfile(output_folder + '/metrics.txt'):
    time.sleep(1)
  with open(output_folder + '/metrics.txt', 'r') as file:
    return [line.strip() for line in file.readlines()]


if __name__ == "__main__":
  output_folder, data_folder, model_folder = sys.argv[1], sys.argv[2], sys.argv[3]
  model = wait_for_model(model_folder)
  model_name = model_folder.split('/')[-1]
  test_data = wait_for_data(data_folder)
  functions = read_evaluation_functions(output_folder)

  X = test_data.drop('Survived', axis=1)
  y_true = test_data['Survived']
  y_pred = model.predict(X)
  results = dict()

  for function in functions:
    if function == 'accuracy':
      results['accuracy'] = accuracy_score(y_true, y_pred)
    elif function == 'precision':
      results['precision'] = precision_score(y_true, y_pred, average='weighted')
    elif function == 'f1':
      results['f1'] = f1_score(y_true, y_pred, average='weighted')
    elif function == 'recall':
      results['recall'] = recall_score(y_true, y_pred, average='weighted')
    elif function == 'r2':
      results['r2'] = r2_score(y_true, y_pred)
    elif function == 'mse':
      results['mse'] = mean_squared_error(y_true, y_pred)
    elif function == 'rmse':
      results['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
    elif function == 'mae':
      results['mae'] = mean_absolute_error(y_true, y_pred)

  with open(output_folder + '/' + model_name + '_evaluation.txt', 'w') as f:
    for key, value in results.items():
      f.write(f"{key}: {value}\n")
