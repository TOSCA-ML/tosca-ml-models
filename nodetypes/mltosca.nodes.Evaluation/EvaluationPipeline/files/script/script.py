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
  file_name = data[1]
  return pd.read_pickle(file_name)


def setup_config_file(project_location, models_fullpath):
  with open(project_location + '/configuration.json', 'w') as file:
    json.dump({'filenames': models_fullpath}, file, indent=4)


def read_evaluation_functions(output_folder: str):
  while not os.path.isfile(output_folder + '/metrics.txt'):
    time.sleep(1)
  with open(output_folder + '/metrics.txt', 'r') as file:
    return [line.strip() for line in file.readlines()]


if __name__ == "__main__":
  project_location, output_folder = sys.argv[1], sys.argv[2]

  while not os.path.isfile(project_location + '/' + output_folder + '/models.txt'):
    time.sleep(1)

  with open(project_location + '/' + output_folder + '/models.txt') as file:
    models = [line.rstrip() for line in file]

  models_fullpath = []

  for model_description in models:
    model_name, data_folder, target = model_description.split("#_#")

    model = wait_for_model(project_location + '/' + model_name)
    test_data = wait_for_data(project_location + '/' + data_folder)
    functions = read_evaluation_functions(project_location + '/' + output_folder)

    X = test_data.drop(target, axis=1)
    y_true = test_data[target]
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

    fullpath = project_location + '/' + output_folder + '/' + model_name + '.txt'
    models_fullpath.append(fullpath)
    with open(fullpath, 'w') as f:
      for key, value in results.items():
        f.write(f"{key}: {value}\n")

  setup_config_file(project_location + '/' + output_folder, models_fullpath)
