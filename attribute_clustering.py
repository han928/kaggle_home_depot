from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np
import scipy as sp
import zipfile

df_attr = pd.read_csv(zipfile.ZipFile('data/attributes.csv.zip', mode='r').open('attributes.csv'))
# df_desc = pd.read_csv(zipfile.ZipFile('data/product_descriptions.csv.zip', mode='r').open('product_descriptions.csv'))

df_rel = pd.read_csv(zipfile.ZipFile('data/train.csv.zip', mode='r').open('train.csv'), 
                     index_col=0,
                     dtype={'product_uid':   int, 
                            'product_title': str, 
                            'search_term':   str,
                            'relevance':     float})
                            
df_attr.dropna(inplace=True)
df_attr.drop_duplicates(['product_uid', 'name'], keep='first', inplace=True)

unique_ids     = sorted(df_attr.product_uid.unique())
unique_brands  = sorted(df_attr.name.unique())

row_map = pd.Series(xrange(len(unique_ids   )), index=unique_ids)
col_map = pd.Series(xrange(len(unique_brands)), index=unique_brands)

X = sp.sparse.lil_matrix((len(row_map), len(col_map)), dtype=bool)
X[row_map[df_attr.product_uid].values, col_map[df_attr.name].values] = 1
print X.nnz

model = KMeans(n_clusters=4, n_jobs=-1)
model.fit_predict(X)
model.cluster_centers_

df_train, df_test = train_test_split(df_rel, test_size=0.2, random_state=40)

df_high_rel = df_train[df_train.relevance > 2.5]
df_high_rel.loc[:,'row_id'] = row_map[df_high_rel.product_uid.values].fillna(-1).astype(int).values

df_high_rel = df_high_rel[df_high_rel.row_id > -1]
print len(df_high_rel)

df_high_rel.loc[:, 'cluster'] = model.labels_[df_high_rel.row_id]
df_high_rel.head(30)

grpdf = df_high_rel.groupby('cluster')
