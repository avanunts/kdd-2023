import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

from util.global_variables import storage_path
from util.compress import compress
from util.titles import get_titles_proc
import util.path_helpers as ph

"""
config = {
    v: {} - vectorizer params
    c: {} - compression params
}
"""

vb_dir = os.path.join(storage_path, "vector_bases")
compressed_dir = os.path.join(vb_dir, "compressed")
row_dir = os.path.join(vb_dir, "row")


def build_vector_base(locale, config):
    if os.path.exists(compressed_path(locale, config)):
        return
    do_build_vector_base(locale, config)


def do_build_vector_base(locale, config):
    build_row_bows(locale, config['v'])
    row_bows = get_row_bows(locale, config['v'])
    compressed_bows = compress(row_bows, config['c'])
    path = compressed_path(locale, config)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    np.save(path, compressed_bows)


def build_row_bows(locale, v_config):
    if os.path.exists(row_path(locale, v_config)):
        return
    do_build_row_bows(locale, v_config)


def do_build_row_bows(locale, v_config):
    titles = get_titles_proc(locale).title_proc
    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(v_config['ngram']['l'], v_config['ngram']['r'])
    )
    row_bows = vectorizer.fit_transform(titles)
    path = row_path(locale, v_config)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    sparse.save_npz(path, row_bows)


def get_row_bows(locale, v_config):
    return sparse.load_npz(row_path(locale, v_config))


def get_vb(locale, config):
    return get_index(locale, config), get_compressed(locale, config)


def get_index(locale, config):
    titles = get_titles_proc(locale)
    match config['c']['name']:
        case 'fh':
            return titles.id.values
        case 'svd':
            title_len = titles.title_proc.apply(len).values
            return titles.id.values[title_len > config['c']['min_title_len']]
        case other:
            raise NotImplementedError('{} is not a valid option for compressor name (c.name)'.format(other))


def get_compressed(locale, config):
    return np.load(compressed_path(locale, config))


def row_path(locale, v_config):
    return os.path.join(row_dir, ph.config_path(v_config), '{}.npz'.format(locale))


def compressed_path(locale, config):
    return os.path.join(compressed_dir, ph.config_path(config), '{}.npy'.format(locale))
