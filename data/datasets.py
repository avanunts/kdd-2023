import os

import pandas as pd

data_dir = os.path.join("/", "Users", "fridrihkakiev", "Downloads", "kdd 2023 datasets")
sessions_train_path = os.path.join(data_dir, "sessions_train.csv")
products_train_path = os.path.join(data_dir, "products_train.csv")
sessions_test_path = os.path.join(data_dir, "sessions_test_task1.csv")


def convert_prev_items(x):
    remove_characters = {ord(x): None for x in '\n\''}
    return x.strip('[]').translate(remove_characters).split(' ')


def get_sessions_train(nrows=None):
    return pd.read_csv(
        sessions_train_path,
        dtype={'next_item': 'string', 'locale': 'string'},
        converters={'prev_items': convert_prev_items},
        nrows=nrows
    )

def get_sessions_test(nrows=None):
    return pd.read_csv(
        sessions_test_path,
        dtype={'locale': 'string'},
        converters={'prev_items': convert_prev_items},
        nrows=nrows
    )

def get_products_train():
    return pd.read_csv(
        products_train_path,
        dtype={
            'id': 'string', 'locale': 'string', 'title': 'string',
            'brand': 'string', 'color': 'string', 'size': 'string',
            'model': 'string', 'material': 'string', 'author': 'string',
            'desc': 'string'
        },
    )
