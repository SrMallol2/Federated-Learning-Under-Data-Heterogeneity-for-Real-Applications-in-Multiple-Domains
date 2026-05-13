import pickle

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from data.FlagsRegression.dataset import load_dataset

gmms = {}
distances = {}


# Calculates kl divergence of two gmms
def gmm_kl(gmm_p, gmm_q, n_samples=10 ** 5):
    X, _ = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()


def gmm_js(gmm_p, gmm_q, n_samples=10 ** 5):
    X, _ = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y, _ = gmm_q.sample(n_samples)
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2


def cluster_distance(i_p, i_q):
    p, q = aps[int(i_p[0])], aps[int(i_q[0])]
    print(f'Calculating distance for {int(i_p[0])} and {int(i_q[0])}')
    key = ','.join(sorted([p, q]))
    if key in distances:
        print('Hit the cache!')
        return distances[key]
    distance = gmm_js(gmms[p], gmms[q])
    distances[key] = distance
    return distance


raw_data = load_dataset(path='/home/dsalami/dataset/aggregated/pickle_2019-05-13-on7_2min.pkl', normalize=True)
aps = np.unique(raw_data['AP ID'])
print(f'Total number of APs: {len(aps)}')

for i in tqdm(list(range(len(aps)))):
    original_data = raw_data['Bytes'][raw_data['AP ID'] == aps[i]].to_numpy()
    gmms[aps[i]] = GaussianMixture(n_components=10 if len(original_data) >= 10 else len(original_data),
                                   random_state=0).fit(original_data.reshape(-1, 1))

try:
    distance_matrix = pairwise_distances(np.array(range(len(aps))).reshape(-1, 1), metric=cluster_distance,
                                         n_jobs=20)
    fclust1 = fclusterdata(np.array(range(len(aps))).reshape(-1, 1), 0.1, metric=cluster_distance)
except:
    print('An exception happened!')
finally:
    with open('distances.pkl', 'wb') as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('distances_backup.pkl', 'wb') as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Distances are saved now!')


with open('fclusters.pkl', 'wb') as handle:
    pickle.dump(fclust1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('aps.pkl', 'wb') as handle:
    pickle.dump(aps, handle, protocol=pickle.HIGHEST_PROTOCOL)
