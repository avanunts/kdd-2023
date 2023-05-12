import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def encode_products(products_ds):
    enc = OrdinalEncoder()
    enc.fit(products_ds.id.values.reshape(-1, 1))
    return enc


def get_ith_item_encoded(s1, enc, i):
    return enc.transform(s1.prev_items.apply(lambda x: x[i]).astype('string').values.reshape(-1, 1)).flatten()


def get_next_item_encoded(data, enc):
    return enc.transform(data.next_item.values.reshape(-1, 1)).flatten()


def get_fixed_length_sessions_encoded(sessions_ds, length, enc, with_next_item=False):
    session_length = sessions_ds.prev_items.apply(len)
    sl = sessions_ds[session_length == length]  # sl - sessions of length = length
    columns = {str(i): get_ith_item_encoded(sl, enc, i) for i in range(length)}
    if with_next_item:
        columns[str(length)] = get_next_item_encoded(sl, enc)
    return pd.DataFrame(columns)
