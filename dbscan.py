import pyspark as ps
import sklearn.cluster as skc
from scipy.spatial.distance import *

from partition import KDPartitioner

LOGGING = False


def dbscan_partition(iterable, params):
    """
    :type iterable: iter
    :param iterable: iterator yielding ((key, partition), vector)
    :type params: dict
    :param params: dictionary containing sklearn DBSCAN parameters
    :rtype: iter
    :return: ((key, cluster_id), v)
    Performs a DBSCAN on a given partition of the data
    """
    # read iterable into local memory
    data = list(iterable)
    (key, part), vector = data[0]
    x = np.array([v for (_, __), v in data])
    y = np.array([k for (k, _), __ in data])
    # perform DBSCAN
    model = skc.DBSCAN(**params)
    c = model.fit_predict(x)
    cores = set(model.core_sample_indices_)
    # yield ((key, cluster_id), v), non-core samples labeled with *
    for i in xrange(len(c)):
        flag = '' if i in cores else '*'
        yield ((y[i], '%i:%i%s' % (part, c[i], flag)), x[i])


def map_cluster_id(((key, cluster_id), v), cluster_dict):
    """
    :type cluster_dict: dict
    :param cluster_dict: Dictionary of cluster id mappings
    :rtype: (int, int), numpy.ndarray
    :return: (key, cluster label), vector
    Modifies the item key to include the remapped cluster label,
    choosing the first id if there are multiple ids
    """
    cluster_id = cluster_id.split(',')[0].strip('*')
    if '-1' not in cluster_id and cluster_id in cluster_dict:
        return (key, cluster_dict[cluster_id]), v
    else:
        return (key, -1), v


class DBSCAN(object):
    """
    :eps: nearest neighbor radius
    :min_samples: minimum number of sample within radius eps
    :metric: distance metric
    :max_partitions: maximum number of partitions used by KDPartitioner
    :data: copy of the data used to train the model including
    :result: RDD containing the (key, cluster label) pairs
    :bounding_boxes: dictionary of BoundingBoxes used to partition the
        data
    :expanded_boxes: dictionary of BoundingBoxes expanded by 2 eps in
        all directions, used to partition data
    :neighbors: dictionary of RDD containing the ((key, cluster label),
        vector) for data within each partition
    :cluster_dict: dictionary of mappings for neighborhood cluster ids
        to global cluster ids
    """

    def __init__(self, eps=0.5, min_samples=5, metric=euclidean,
                 max_partitions=None):
        """
        :type eps: float
        :param eps: nearest neighbor radius
        :type min_samples: int
        :param min_samples: minimum number of samples within radius eps
        :type metric: callable
        :param metric: distance metric (should be
            scipy.spatial.distance.euclidian or
            scipy.spatial.distance.cityblock)
        :type max_partitions: int
        :param max_partitions: maximum number of partitions in
            KDPartitioner
        Using a metric other than euclidian or cityblock/Manhattan may
        not work as the bounding boxes expand in such a way that
        other metrics may return distances less than eps for points
        outside the box.
        """
        self.eps = eps
        self.min_samples = int(min_samples)
        self.metric = metric
        self.max_partitions = max_partitions
        self.data = None
        self.result = None
        self.bounding_boxes = None
        self.expanded_boxes = None
        self.neighbors = None
        self.cluster_dict = None

    def train(self, data):
        """
        :type data: pyspark.RDD
        :param data: (key, k-dim vector like)
        Train the model using a (key, vector) RDD
        """
        parts = KDPartitioner(data, self.max_partitions)
        self.data = data
        neighbors = {}
        self.bounding_boxes = parts.bounding_boxes
        self.expanded_boxes = {}
        self._create_neighborhoods()
        # repartition data set on the partition label
        self.data = self.data.map(lambda ((k, p), v): (p, (k, v))) \
            .partitionBy(len(parts.partitions)) \
            .map(lambda (p, (k, v)): ((k, p), v))
        # create parameters for sklearn DBSCAN
        params = {'eps': self.eps, 'min_samples': self.min_samples,
                  'metric': self.metric}
        # perform dbscan on each part
        self.data = self.data.mapPartitions(
            lambda iterable: dbscan_partition(iterable, params))
        self.data.cache()
        self._remap_cluster_ids()

    def assignments(self):
        """
        :rtype: pyspark.RDD
        :return: (key, cluster_id)
        Retrieve the results of the DBSCAN
        """
        return self.result.collect()

    def _create_neighborhoods(self):
        """
        Expands bounding boxes by 2 * eps and creates neighborhoods of
        items within those boxes with partition ids in key.
        """
        neighbors = {}
        new_data = self.data.context.emptyRDD()
        for label, box in self.bounding_boxes.iteritems():
            expanded_box = box.expand(2 * self.eps)
            self.expanded_boxes[label] = expanded_box
            neighbors[label] = self.data.filter(
                lambda (k, v): expanded_box.contains(v)) \
                .map(lambda (k, v): ((k, label), v))
            new_data = new_data.union(neighbors[label])
        self.neighbors = neighbors
        self.data = new_data

    def _remap_cluster_ids(self):
        """
        Scans through the data for collisions in cluster ids, creating
        a mapping from partition level clusters to global clusters
        """
        point_labels = self.data.map(lambda ((k, c), v): (k, c)).groupByKey() \
            .map(lambda (k, c): (k, list(c))).collect()
        new_cluster_label = 0
        cluster_dict = {}
        if LOGGING:
            with open('dbscan.log', 'w') as f:
                f.write('key,clusters')
                for key, cluster_ids in point_labels:
                    f.write('\n%i,%s' % (
                        key,
                        ';'.join(np.array(list(cluster_ids)).astype(str))))
        for k, cluster_ids in point_labels:
            cluster_ids = np.array(list(cluster_ids))
            in_dict = np.array(
                [cluster_id in cluster_dict for cluster_id in cluster_ids])
            if np.any(in_dict):
                # find lowest label for labeled clusters
                labels = [cluster_dict[cluster_id]
                          if cluster_id in cluster_dict else new_cluster_label
                          for cluster_id in cluster_ids]
                label = np.min(labels)
                for key, value in cluster_dict.iteritems():
                    if value in labels and value != label:
                        cluster_dict[key] = label
            else:
                # create/increment label
                label = new_cluster_label
                new_cluster_label += 1
            for cluster_id in cluster_ids:
                # for each cluster_id, excluding noise and non-core samples
                if '-1' not in cluster_id and '*' not in cluster_id:
                    # map that id onto the label
                    cluster_dict[cluster_id] = label
        self.cluster_dict = cluster_dict
        self.result = self.data \
            .map(lambda x: map_cluster_id(x, cluster_dict)) \
            .map(lambda ((k, c), v): (k, c)).reduceByKey(min).sortByKey()
        self.result.cache()


if __name__ == '__main__':
    # Example of pypadis.DBSCAN
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    from time import time
    from itertools import izip
    import os

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers,
                                cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    sc = ps.SparkContext()
    test_data = sc.parallelize(enumerate(X))
    start = time()
    dbscan = DBSCAN(0.3, 10)
    dbscan.train(test_data)
    result = np.array(dbscan.assignments())
    print time() - start
    clusters = result[:, 1]
    temp = dbscan.data.glom().collect()
    colors = cm.spectral(np.linspace(0, 1, len(dbscan.bounding_boxes)))
    if not os.access('plots', os.F_OK):
        os.mkdir('plots')
    for i, t in enumerate(temp):
        x = [t2[1][0] for t2 in t]
        y = [t2[1][1] for t2 in t]
        c = [int(t2[0][1].split(':')[1].strip('*')) for t2 in t]
        l = [int(t2[0][1].split(':')[0]) for t2 in t]
        box1 = dbscan.bounding_boxes[l[0]]
        box2 = dbscan.expanded_boxes[l[0]]
        in_box = [box2.contains([a, b]) for a, b in izip(x, y)]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.add_patch(
            patches.Rectangle(box1.lower, *(box1.upper - box1.lower),
                              alpha=0.4, color=colors[i], zorder=0))
        ax.add_patch(
            patches.Rectangle(box2.lower, *(box2.upper - box2.lower),
                              fill=False, zorder=0))
        plt.scatter(x, y, c=c, zorder=1)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.savefig('plots/partition_%i.png' % i)
        plt.close()
    x = X[:, 0]
    y = X[:, 1]
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(x, y, c=clusters)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig('plots/clusters.png')
    plt.close()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i, b in dbscan.bounding_boxes.iteritems():
        ax.add_patch(
            patches.Rectangle(b.lower, *(b.upper - b.lower),
                              color=colors[i], alpha=0.5, zorder=0))
    plt.scatter(x, y, c=clusters, zorder=1)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig('plots/clusters_partitions.png')
    plt.close()
