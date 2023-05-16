import faiss
import pandas as pd
import numpy as np

from util.vector_base import get_vb


class KnnPredictor:
    def __init__(self, locale, vb_config):
        self.locale = locale
        ids, vectors = get_vb(locale, vb_config)
        self.ids = ids.astype('str')
        self.id_to_vec = pd.DataFrame(
            {'last_item': ids, 'vector': vectors.tolist()}
        )
        self.id_to_vec['last_item'] = self.id_to_vec.last_item.astype('string')
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    def to_vectors(self, queries):
        last_item = pd.DataFrame({'last_item': queries.prev_items.apply(lambda x: x[-1]).astype('string')})
        print(last_item.shape)
        item_to_vec = last_item.merge(self.id_to_vec, on='last_item', suffixes=('_l', '_r'))
        print(item_to_vec.shape)
        return np.array(item_to_vec.vector.values.tolist())

    def to_id(self, i):
        return self.ids[i]

    def to_ids(self, indices):
        vectorized = np.vectorize(self.to_id)
        return vectorized(indices).tolist()

    def predict(self, queries, nn):
        vectors_q = self.to_vectors(queries)
        print(vectors_q.shape)
        _, indices = self.index.search(vectors_q, nn)
        print(indices.shape)
        queries['next_item_prediction'] = self.to_ids(indices)
