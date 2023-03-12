import os
import sys
import json
import shutil
import pandas as pd
from pandas import DataFrame


def setup_config_file(project_location, dataframe_name='df.pkl'):
  with open(project_location + '/configuration.json', 'w') as file:
    json.dump({'filenames': [project_location + '/' + dataframe_name]}, file, indent=4)


def read_csv(filepath: str, parameters: str) -> DataFrame:
    delimiter, encoding = parameters.split('#_#')
    return pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)


def save_dataframe(df: DataFrame, project_location: str):
    df.to_pickle(project_location + '/processing.pkl')
    os.rename(project_location + '/processing.pkl', project_location + '/df.pkl')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit('Not enough args')
    project_location, filepath, parameters = sys.argv[1], sys.argv[2], sys.argv[3]
    setup_config_file(project_location)
    if filepath.endswith('.csv'):
      df = read_csv(filepath, parameters)
      save_dataframe(df, project_location)
    else:
      shutil.copy(filepath, project_location + '/df.pkl')
