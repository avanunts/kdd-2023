import os

import pandas as pd

from util.global_variables import storage_path


# titles are obtained in process-titles.ipynnb, TODO: move code here


titles_dir = os.path.join(storage_path, 'titles')


def save_titles_proc(locale, ds):
    ds.to_csv(titles_proc_path(locale), index=False)


def get_titles_proc(locale):
    return pd.read_csv(titles_proc_path(locale))


def titles_proc_path(locale):
    return os.path.join(titles_dir, '{}.csv'.format(locale))