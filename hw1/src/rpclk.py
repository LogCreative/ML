import copy
import numpy as np
import matplotlib.pyplot as plt
from datagen import GMMData


class RPCLK:
    def __init__(self, dataset, cluster_num, gamma, random_seed=None):
        """RPCL refined k-mean algorithm.

        Args:
            dataset (numpy.array): the dataset generated from datagen.GMMData
            cluster_num (int): the maximum number of clusters
            gamma (float): the running away factor for rival cluster.
            random_seed (int, optional): The random seed. Defaults to None.
        """
        self.dataset = dataset
        self.cluster_num = cluster_num
        self.gamma = gamma
        self.random_seed = random_seed

    def init_means(self):
        """Initialize means point.

        Returns:
            numpy.array: the initialized means.
        """
        self.dim = np.shape(self.dataset)[1]
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        means = np.random.rand(self.cluster_num, self.dim)
        bounding_max = np.array([np.max(self.dataset, 0)]).repeat(
            self.cluster_num, axis=0
        )
        bounding_min = np.array([np.min(self.dataset, 0)]).repeat(
            self.cluster_num, axis=0
        )
        means = bounding_min + (bounding_max - bounding_min) * means
        np.random.seed()
        return means

    def contribution(self, means):
        """Calculate the contribution for each data point and cluster.

        Args:
            means (numpy.array): the mean points at this iteration.

        Returns:
            numpy.array: the contribution array for cluster, datapoint. 1 stands for belonging and -self.gamma stands for rival, 0 stands for nothing.
        """
        contrib = np.zeros((len(means), self.dataset.shape[0]))
        for i, point in enumerate(self.dataset):
            sqrdist = np.sum(
                np.square(np.array([point]).repeat(len(means), axis=0) - means),
                axis=1,
            )

            c = np.argmin(sqrdist)
            contrib[c][i] = 1

            sqrdist[c] = np.inf
            r = np.argmin(sqrdist)
            contrib[r][i] = -self.gamma
        return contrib

    def iteration(self, means, contrib, lr):
        """Iterate mean points.

        Args:
            means (numpy.array): the original mean points.
            contrib (numpy.array): the contribution array generated from contribution()
            lr (float): learning rate at this iteration.

        Returns:
            float: the changed difference before and after the iteration.
        """
        contrib_one = np.clip(contrib, 0, 1)
        contrib_neg = np.clip(contrib, -self.gamma, 0)
        dataset_size = self.dataset.shape[0]
        dim = self.dataset.shape[1]

        stale_k = []
        old_means = copy.deepcopy(self.means)

        for k in range(len(means)):
            if np.sum(contrib_one[k]) != 0:
                means[k] = np.average(self.dataset, weights=contrib_one[k], axis=0)
                means[k] += (
                    np.sum(
                        (
                            self.dataset
                            - np.array([means[k]]).repeat(dataset_size, axis=0)
                        )
                        * np.array([contrib_neg[k]]).repeat(dim, axis=0).T,
                        axis=0,
                    )
                    * lr
                )
            else:
                stale_k.append(k)

        delta = np.sum(np.sum(np.square(self.means - old_means), axis=1))

        self.means = np.delete(means, stale_k, 0)
        self.contrib = np.delete(contrib, stale_k, 0)
        return delta

    def isStopping(self, delta):
        """Early stopping criteria for delta.

        Args:
            delta (float): the difference on mean points.

        Returns:
            bool: stop or not.
        """
        self.deltas.append(delta)
        if len(self.deltas) > 1:
            if np.abs(self.deltas[len(self.deltas) - 2] - delta) <= 0.01:
                self.stale_step += 1
                if self.stale_step > 5:
                    return True
            else:
                self.stale_step = 0
        return False

    def train(self, visual=False):
        """train the model.

        Args:
            visual (bool, optional): Visualize every step or not. Defaults to False.
        """
        self.means = self.init_means()

        self.deltas = []
        self.stale_step = 0
        lr = 0.01 * self.cluster_num

        if visual:
            self.visualize_init()

        while True:
            prev_cluster_num = len(self.means)
            self.contrib = self.contribution(self.means)
            delta = self.iteration(self.means, self.contrib, lr)
            if self.isStopping(delta):
                break
            if visual:
                self.visualize_hook()
            if len(self.means) != prev_cluster_num:  # For early changes
                lr *= len(self.means) / self.cluster_num
            if len(self.deltas) % len(self.means) == 0:
                lr /= 2  # stair decay

    def visualize(self, ax):
        """Visualize clusters.

        Args:
            ax (pyplot.axes.Axes): the axes to be drawn.
        """
        for k in range(len(self.means)):
            cluster_points = self.dataset[np.where(self.contrib[k] == 1)]
            ax.plot(
                cluster_points[:, 0], cluster_points[:, 1], ".", color=plt.cm.tab20(k)
            )

    def visualize_center(self, ax):
        """Visualize center points (mean points).

        Args:
            ax (pyplot.axes.Axes): the axes to be drawn.
        """
        for k in range(len(self.means)):
            ax.plot(self.means[:, 0], self.means[:, 1], "x", color="red")

    def visualize_delta(self, ax):
        """Visualize delta tendency.

        Args:
            ax (pyplot.axes.Axes): the axes to be drawn.
        """
        ax.plot(self.deltas)

    def visualize_init(self):
        """Initialize visualization for interactive traing."""
        plt.ion()
        self.train_fig = plt.figure()
        self.train_ax = plt.axes()

    def visualize_hook(self):
        """The visualization hook for iteractive training."""
        self.train_ax.clear()
        self.visualize(self.train_ax)
        self.visualize_center(self.train_ax)
        self.train_ax.set_title(
            "#%2d delta = %.2f K=%2d"
            % (len(self.deltas), self.deltas[len(self.deltas) - 1], len(self.means))
        )
        self.train_fig.savefig(
            ("" if self.random_seed is None else str(self.random_seed))
            + "-"
            + str(len(self.deltas))
        )
        plt.pause(0.5)


if __name__ == "__main__":
    gmmdata = GMMData(cluster_num=3, dim=2, sample_size=1000, random_seed=20)
    dataset = gmmdata.get_data()
    rpclk = RPCLK(dataset, cluster_num=12, gamma=0.1, random_seed=30)
    rpclk.train(True)

    # fig = plt.figure()
    # ax = plt.axes()  # projection='3d' is required if you use 3-D data.
    # rpclk.visualize(ax)
    # plt.show()

    # fig = plt.figure()
    # ax = plt.axes()
    # rpclk.visualize_delta(ax)
    # plt.show()
