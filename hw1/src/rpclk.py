import copy
import numpy as np
import matplotlib.pyplot as plt
from datagen import GMMData


class RPCLK:
    def __init__(self, dataset, cluster_num, gamma):
        self.dataset = dataset
        self.cluster_num = cluster_num
        self.gamma = gamma

    def init_means(self, random_seed=None):
        self.dim = np.shape(self.dataset)[1]
        if random_seed is not None:
            np.random.seed(random_seed)
        means = np.random.rand(self.cluster_num, self.dim)
        bounding_max = np.array([np.max(self.dataset, 0)]).repeat(
            self.cluster_num, axis=0
        )
        bounding_min = np.array([np.min(self.dataset, 0)]).repeat(
            self.cluster_num, axis=0
        )
        means = bounding_min + (bounding_max - bounding_min) * means
        return means

    def contribution(self, means):
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
        self.deltas.append(delta)
        if len(self.deltas) > 1:
            if np.abs(self.deltas[len(self.deltas) - 2] - delta) <= 0.01:
                self.stale_step += 1
                if self.stale_step > 5:
                    return True
            else:
                self.stale_step = 0
        return False

    def train(self, visual=False, random_seed=None):
        self.means = self.init_means(random_seed)

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
        for k in range(len(self.means)):
            cluster_points = self.dataset[np.where(self.contrib[k] == 1)]
            ax.plot(
                cluster_points[:, 0], cluster_points[:, 1], ".", color=plt.cm.tab20(k)
            )

    def visualize_center(self, ax):
        for k in range(len(self.means)):
            ax.plot(self.means[:, 0], self.means[:, 1], "x", color="red")

    def visualize_delta(self, ax):
        ax.plot(self.deltas)

    def visualize_init(self):
        plt.ion()
        self.train_fig = plt.figure()
        self.train_ax = plt.axes()

    def visualize_hook(self):
        self.train_ax.clear()
        self.visualize(self.train_ax)
        self.visualize_center(self.train_ax)
        self.train_ax.set_title(
            "#%2d delta = %.2f K=%2d"
            % (len(self.deltas), self.deltas[len(self.deltas) - 1], len(self.means))
        )
        plt.pause(0.5)


if __name__ == "__main__":
    gmmdata = GMMData(cluster_num=3, dim=2, sample_size=1000, random_seed=20)
    dataset = gmmdata.get_data()
    rpclk = RPCLK(dataset, cluster_num=3, gamma=0.05)
    rpclk.train(True)

    # fig = plt.figure()
    # ax = plt.axes()  # projection='3d' is required if you use 3-D data.
    # rpclk.visualize(ax)
    # plt.show()

    # fig = plt.figure()
    # ax = plt.axes()
    # rpclk.visualize_delta(ax)
    # plt.show()
