from sklearn.feature_extraction import FeatureHasher
from scipy import sparse


"""
config = {
    name: "..." - compression algorithm name
    ...
}
"""


def compress(sparse_vectors, config):
    match config['name']:
        case 'fh':
            return compress_fh(sparse_vectors, config)
        case other:
            raise NotImplementedError('{} is not among valid options for c_alg_name'.format(other))


def compress_fh(sparse_vectors, config):
    h = FeatureHasher(n_features=config['n_features'])
    inv_norms = (1 / sparse_vectors.power(2).sum(axis=1)).A.flatten() ** 0.5
    normalized_vectors = multiply_by_dense(sparse_vectors, inv_norms)
    dicts = sparse_matrix_to_dicts(normalized_vectors)
    return h.transform(dicts).toarray()


def row_to_dict(row):
    d = dict()
    for item in row:
        i, val = str(item[0][1]), item[1]
        d[i] = val
    return d


def sparse_matrix_to_dicts(matrix):
    dicts = []
    for i in range(matrix.shape[0]):
        row = matrix[i].todok().items()
        new_dict = row_to_dict(row)
        dicts.append(new_dict)
    return dicts

def multiply_by_dense(a, b):
    return sparse.diags(b) * a
