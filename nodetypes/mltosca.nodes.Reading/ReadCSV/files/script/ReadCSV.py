import os
import sys
import pandas as pd
from pandas import DataFrame


def read_csv(filepath, delimiter=',', encoding='UTF-8') -> DataFrame:
    return pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)


def save_dataframe(df: DataFrame):
    path = os.path.expanduser('~') + '/mltosca'
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_pickle(path + '/df.pkl')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Not enough args")
    filepath, delimiter, encoding = sys.argv[1], sys.argv[2], sys.argv[3]
    df = read_csv(filepath, delimiter, encoding)
    save_dataframe(df)
