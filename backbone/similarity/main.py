import hnswlib
import numpy as np

dim = 512
p = hnswlib.Index(space='cosine', dim=dim)
p.init_index(max_elements=100000, ef_construction=200, M=16)

data = np.random.rand(100000, dim)
p.add_items(data)

labels, distances = p.knn_query(data[:1], k=5)