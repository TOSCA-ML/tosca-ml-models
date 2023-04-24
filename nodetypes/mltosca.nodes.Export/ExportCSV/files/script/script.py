import os
import sys
import time
import json
import pickle
import pandas as pd


def read_config_file(config_path: str) -> str:
  with open(config_path, 'r') as f:
    config = json.load(f)
  return config['filenames']


def wait_for_data(data_folder: str):
  wait_until_file_created(data_folder + '/configuration.json')
  return read_config_file(data_folder + '/configuration.json')


def wait_until_file_created(filename):
  while not os.path.isfile(filename):
    time.sleep(1)


if __name__ == "__main__":
  project_location, output_folder, export_folder = sys.argv[1], sys.argv[2], sys.argv[3]

  filenames = wait_for_data(project_location + '/' + output_folder)

  for filename in filenames:
    wait_until_file_created(filename)
    name = filename.split('/')[-1].split('.')[0]
    with open(filename, 'rb') as f:
      data = pickle.load(f)
    df = pd.DataFrame(data)
    df.to_csv(export_folder + '/' + output_folder + '_ ' + name + '.csv')
