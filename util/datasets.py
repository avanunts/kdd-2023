import os

import pandas as pd

from util.global_variables import storage_path

datasets_dir_path = os.path.join(storage_path, "datasets")


def sessions_path(locale, typ):
    return os.path.join(datasets_dir_path, '{}_sessions_{}.parquet'.format(locale, typ))


def products_path(locale):
    return os.path.join(datasets_dir_path, '{}_products.parquet'.format(locale))


def get_sessions(locale, typ):
    return pd.read_parquet(
        sessions_path(locale, typ),
    )


def get_products(locale):
    return pd.read_parquet(products_path(locale))

