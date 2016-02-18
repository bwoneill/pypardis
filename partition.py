from Queue import Queue
from geometry import BoundingBox
from operator import add
import numpy as np
import pyspark as ps
from random import randint


def map_results((key, vector), broadcast_dict):
    """
    :type key: int
    :param key:
    :type vector: numpy,ndarray
    :param vector
    :type broadcast_dict: pyspark.Broadcast
    :param broadcast_dict:
    :return:
    """
    for p, box in broadcast_dict.value.iteritems():
        if box.contains(vector):
            return (key, p), vector


class KDPartitioner(object):
    """
    :bounding_boxes: dictionary of int => BoundingBox for that
        partition label
    :k: dimensionality of the data
    :max_partitions: maximum number of partitions
    :persist: Spark persistence/storage level
    """

    def __init__(self, data, max_partitions=None, k=None, persist=None):
        """
        :type data: pyspark.RDD
        :param data: pyspark RDD (key, k-dim vector like)
        :type max_partitions: int
        :param max_partitions: maximum number of partition to split
            into
        :type k: int
        :param k: dimensionality of the data
        :type persist: pyspark.StorageLevel
        :param persist: storage level
        Split a given data set into approximately equal sized partition
        (if max_partitions is a power of 2 ** k) using binary tree
        methods
        """
        self.partitions = {}
        self.old_partitions = {}
        self.persist = persist
        self.data = data
        self.result = None
        self.k = int(k) if k is not None else len(data.first()[1])
        self.max_partitions = int(
            max_partitions) if max_partitions is not None else 4 ** self.k
        self._create_partitions()
        self.id = randint(0, 9999)

    def _create_partitions(self):
        """
        """
        box = self.data.aggregate(BoundingBox(k=self.k),
                                  lambda total, (_, v): total.union(
                                      BoundingBox(v)),
                                  lambda total, v: total.union(v))
        todo_q = Queue()
        todo_q.put(0)
        done_q = Queue()
        self.bounding_boxes = {0: box}
        if self.persist is not None:
            self.partitions[0] = self.data
        next_label = 1
        while next_label < self.max_partitions:
            if not todo_q.empty():
                current_label = todo_q.get()
                box1, box2 = self._min_var_split(current_label)
                if self.persist is not None:
                    self._persist_partitions(box1, box2, current_label,
                                             next_label)
                self.bounding_boxes[current_label] = box1
                self.bounding_boxes[next_label] = box2
                done_q.put(current_label)
                done_q.put(next_label)
                next_label += 1
            else:
                todo_q = done_q
                done_q = Queue()

    def _min_var_split(self, index):
        """
        :type index: int
        :param index: bounding index
        :rtype: BoundingBox, BoundingBox
        :return: tuple of bounding boxes split from the bounding box
            with the given index
        Split the given partition into equal sized partitions along the
        axis with greatest variance.
        """
        box = self.bounding_boxes[index]
        if self.persist is None:
            partition = self.data.filter(lambda (_, v): box.contains(v))
        else:
            partition = self.partitions[index]
        k = self.k
        moments = partition.aggregate(np.zeros((3, k)),
                                      lambda x, (keys, vector): x + np.array(
                                          [np.ones(k), vector,
                                           vector ** 2]), add)
        means = moments[1] / moments[0]
        variances = moments[2] / moments[0] - means ** 2
        axis = np.argmax(variances)
        std_dev = np.sqrt(variances[axis])
        bounds = np.array(
            [means[axis] + (i - 3) * 0.3 * std_dev for i in xrange(7)])
        counts = partition.aggregate(np.zeros(7),
                                     lambda x, (_, v):
                                     x + 2 * (v[axis] < bounds) - 1,
                                     add)
        counts = np.abs(counts)
        boundary = bounds[np.argmin(counts)]
        return box.split(axis, boundary)

    def _persist_partitions(self, box1, box2, current_label, next_label):
        """
        :type box1: geometry.BoundingBox
        :param box1:
        :type box2: geometry.BoundingBox
        :param box2:
        :type current_label: int
        :param current_label:
        :type next_label: int
        :param next_label:
        """
        if current_label in self.old_partitions:
            self.old_partitions[current_label].unpersist()
            del self.old_partitions[current_label]
        p = self.partitions[current_label]
        self.old_partitions[current_label] = p
        p1 = p.filter(lambda (_, v): box1.contains(v))
        p1.setName('KDPartitioner-%i-%i-%i' % (self.id, randint(0, 9999),
                                               current_label))
        p1.persist(self.persist)
        self.partitions[current_label] = p1
        p2 = p.filter(lambda (_, v): box2.contains(v))
        p2.setName('KDPartitioner-%i-%i-%i' % (self.id, randint(0, 9999),
                                               next_label))
        p2.persist(self.persist)
        self.partitions[next_label] = p2

    def get_results(self):
        """

        :return:
        """
        if self.result is None:
            broadcast_var = self.data.context.broadcast(self.bounding_boxes)
            self.result = self.data.map(
                lambda x: map_results(x, broadcast_var))
        return self.result

    def unpersist(self):
        """
        Unpersist all stored partitions.
        """
        for p in self.partitions.itervalues():
            p.unpersist()
            del p
        for p in self.old_partitions.itervalues():
            p.unpersist()
            del p


if __name__ == '__main__':
    # Example of partition.KDPartition
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    from time import time
    import os

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers,
                                cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    sc = ps.SparkContext()
    test_data = sc.parallelize(enumerate(X))
    start = time()
    kdpart = KDPartitioner(test_data, 16, 2)
    final = kdpart.get_results().collect()
    print 'Total time:', time() - start
    partitions = [a[0][1] for a in final]
    x = [a[1][0] for a in final]
    y = [a[1][1] for a in final]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    colors = cm.spectral(np.linspace(0, 1, len(kdpart.bounding_boxes)))
    for label, box in kdpart.bounding_boxes.iteritems():
        ax.add_patch(
            patches.Rectangle(box.lower, *(box.upper - box.lower),
                              alpha=0.5, color=colors[label], zorder=0))
    plt.scatter(x, y, c=partitions, zorder=1)
    if not os.access('plots', os.F_OK):
        os.mkdir('plots')
    plt.savefig('plots/partitioning.png')
    plt.close()
    plt.scatter(x, y)
    plt.savefig('plots/toy_data.png')
    plt.close()
