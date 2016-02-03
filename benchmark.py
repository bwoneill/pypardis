import pyspark as ps
from dbscan import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from time import time
import numpy as np
import sys

if __name__ == '__main__':
    i = int(sys.argv[1])
    centers = [[1, 1], [-1, -1], [1, -1]]
    samples = [750, 7500, 75000, 750000, 7500000]
    eps = [0.3, 0.1, 0.03, 0.01, 0.003]
    n_part = [16, 128, 1024, 8192, 65536]
    sc = ps.SparkContext()
    X, labels_true = make_blobs(n_samples=samples[i], centers=centers,
                                cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    test_data = sc.parallelize(enumerate(X))
    start = time()
    dbscan = DBSCAN(eps[i], 10, max_partitions=n_part[i])
    dbscan.train(test_data)
    result = np.array(dbscan.assignments())
    run_time = time() - start
    with open('benchmark.csv', 'w') as f:
        f.write('\n%i,%f,%i,%i' % (samples[i], eps[i], n_part[i], run_time))
