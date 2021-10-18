from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.spatial import distance
import numpy as np
import cv2

class K_MEAS:
    def __init__(self, file_path, embeddings):
        self.data = {'file_path': file_path, 'embeddings': embeddings, 'cluster_result': None}
        self.k_start = None

    def search_num_cluster(self, encodings, k_min, k_max, random_stat):
        kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 500,"random_state": random_stat}
        sse = [] # A list holds the SSE values for each k
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(encodings)
            sse.append(kmeans.inertia_)
        # Use kneed's KneeLocator to select elbow point
        kl = KneeLocator(range(k_min, k_max + 1), sse, curve="convex", direction="decreasing")
        self.k_start = kl.elbow

    def cluster(self, encodings, k, random_stat):
        kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=random_stat)
        kmeans.fit(encodings)
        self.data['cluster_result'] = kmeans

    def cluster_filter(self, cluster_result, cos_dis_thr, resize_img_dim):
        black_img = np.full((resize_img_dim[1],resize_img_dim[0], 3), (0, 0, 0), np.uint8)
        res = {i: [] for i in range(self.k_start)}
        for path, embedding, cid in zip(self.data['file_path'], \
                                        self.data['embeddings'], \
                                        self.data['cluster_result'].labels_):
            dis = distance.cosine(embedding, self.data['cluster_result'].cluster_centers_[cid])
            img = cv2.resize(cv2.imread(path), (260, 195))
            if dis >= cos_dis_thr:
                img = cv2.addWeighted(img, 0.3, black_img, 0.7, 0)
            cv2.putText(img, '{:.2f}'.format(dis), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
            res[cid].append([dis, img])
        return res