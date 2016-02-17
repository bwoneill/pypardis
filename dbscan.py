import pyspark as ps
import sklearn.cluster as skc
from scipy.spatial.distance import *
from partition import KDPartitioner
from aggregator import ClusterAggregator
from operator import add

LOGGING = False


def dbscan_partition(iterable, params):
    """
    :type iterable: iter
    :param iterable: iterator yielding (partition, (key, vector))
    :type params: dict
    :param params: dictionary containing sklearn DBSCAN parameters
    :rtype: iter
    :return: ((key, cluster_id), v)
    Performs a DBSCAN on a given partition of the data
    """
    # read iterable into local memory
    data = list(iterable)
    part = data[0][0]
    x = np.array([v for p, (k, v) in data])
    y = np.array([k for p, (k, v) in data])
    # perform DBSCAN
    model = skc.DBSCAN(**params)
    c = model.fit_predict(x)
    cores = set(model.core_sample_indices_)
    # yield (key, cluster_id), non-core samples labeled with *
    for i in xrange(len(c)):
        flag = '' if i in cores else '*'
        yield (y[i], '%i:%i%s' % (part, c[i], flag))


def map_cluster_id((key, cluster_id), broadcast_dict):
    """
    :type broadcast_dict: pyspark.Broadcast
    :param broadcast_dict: Broadcast variable containing a dictionary
        of cluster id mappings
    :rtype: int, int
    :return: key, cluster label
    Modifies the item key to include the remapped cluster label,
    choosing the first id if there are multiple ids
    """
    cluster_id = next(iter(cluster_id)).strip('*')
    cluster_dict = broadcast_dict.value
    if '-1' not in cluster_id and cluster_id in cluster_dict:
        return key, cluster_dict[cluster_id]
    else:
        return key, -1


def inflate_events((key, vector), broadcast_dict):
    """
    :type key: int
    :param key: unique identifier for each event
    :type vector: numpy.ndarray
    :param vector:
    :type broadcast_dict: pyspark.Broadcast
    :param broadcast_dict: Broadcast variable containing bounding boxes
    :rtype: list
    :return:
    """
    return [(p, (key, vector)) for p, box in broadcast_dict.value.iteritems()
            if box.contains(vector)]


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
    :cluster_dict: dictionary of mappings for neighborhood cluster ids
        to global cluster ids
    :persist: Spark persistence/storage level
    """

    def __init__(self, eps=0.5, min_samples=5, metric=euclidean,
                 max_partitions=None, persist=None):
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
        self.cluster_dict = None
        self.persist = persist

    def train(self, data):
        """
        :type data: pyspark.RDD
        :param data: (key, k-dim vector like)
        Train the model using a (key, vector) RDD
        """
        parts = KDPartitioner(data, self.max_partitions, persist=self.persist)
        if self.persist is not None:
            parts.unpersist()
        self.data = data
        self.bounding_boxes = parts.bounding_boxes
        self.expanded_boxes = {}
        self._create_neighborhoods()
        # repartition data set on the partition label
        self.data = self.data.partitionBy(parts.max_partitions)
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
        :rtype: list
        :return: list of (key, cluster_id)
        Retrieve the results of the DBSCAN
        """
        return self.result.collect()

    def _create_neighborhoods(self):
        """
        Expands bounding boxes by 2 * eps and creates neighborhoods of
        items within those boxes with partition ids in key.
        """
        expanded_boxes = {}
        for label, box in self.bounding_boxes.iteritems():
            expanded_boxes[label] = box.expand(2 * self.eps)
        broadcast_var = self.data.context.broadcast(expanded_boxes)
        self.data = self.data.flatMap(
            lambda x: inflate_events(x, broadcast_var))
        self.expanded_boxes = expanded_boxes
        self.data.cache()

    def _remap_cluster_ids(self):
        """
        Scans through the data for collisions in cluster ids, creating
        a mapping from partition level clusters to global clusters.
        """
        labeled_points = self.data.groupByKey()
        labeled_points.cache()
        mapper = labeled_points.aggregate(ClusterAggregator(), add, add)
        b_mapper = self.data.context.broadcast(mapper.fwd)
        self.result = labeled_points \
            .map(lambda x: map_cluster_id(x, b_mapper)) \
            .sortByKey()
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
        x = [X[t2[0]][0] for t2 in t]
        y = [X[t2[0]][1] for t2 in t]
        c = [int(t2[1].split(':')[1].strip('*')) for t2 in t]
        l = [int(t2[1].split(':')[0]) for t2 in t]
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
