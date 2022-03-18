import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class GMMData:
    def __init__(
        self,
        cluster_num=3,
        dim=2,
        sample_size=300,
        random_seed=None,
    ):
        """Initialize GMM Model hyper-parameters. Along with initialized dataset.

        Args:
            cluster_num (int, optional): The number of clusters. Defaults to 3.
            dim (int, optional): The dimensionality of data. Defaults to 2.
            sample_size (int, optional): The total number of sampling points. Defaults to 300.
            random_seed (int, optional): The random seed for numpy. Defaults to None.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.cluster_num = cluster_num
        self.dim = dim
        self.sample_size = sample_size
        self.param_gen()
        self.data_gen()

    def param_gen(self):
        """Refresh the parameters for weight, mean and covariance of different clusters."""
        self.weight_list = np.random.rand(self.cluster_num)
        weight_sum = np.sum(self.weight_list)
        self.weight_list /= weight_sum

        self.mean_list = []
        self.cov_list = []
        for k in range(self.cluster_num):
            # FIXME: bad seed may cause the center point too close!
            self.mean_list.append(np.random.rand(self.dim) * self.cluster_num * 2)
            sample_arr = np.random.rand(self.dim, (self.cluster_num + 1))
            self.cov_list.append(np.cov(sample_arr))

    def data_gen(self):
        """Refresh dataset based on the parameters generated in param_gen()"""
        sample_nums = np.random.multinomial(self.sample_size, self.weight_list, 1)[0]
        cluster_samples = []
        for k in range(self.cluster_num):
            cluster_samples.append(
                np.random.multivariate_normal(
                    self.mean_list[k], self.cov_list[k], sample_nums[k]
                )
            )
        self.data = np.concatenate(cluster_samples)

    def get_data(self):
        """Get the dataset.

        Returns:
            np.array: the dataset array, coordinates arranged horizontally.
        """
        return self.data

    def visualize(self, ax):
        """Visualize the dataset. Notice open projection='3d' for 3-D axes.

        Args:
            ax (plt.axes.Axes): The axes it plots.

        Raises:
            ValueError: The dimension is bigger than 3.
            NotImplementedError: The visualization for 1-D array is not implemented yet.
        """
        if self.dim > 3:
            raise ValueError("The dimension is beyond human's comprehension!")
        if self.dim == 1:
            raise NotImplementedError("1-D array is not suitable for visualization!")
        elif self.dim == 2:
            ax.plot(self.data[:, 0], self.data[:, 1], ".", alpha=0.5)
        elif self.dim == 3:
            ax.plot3D(self.data[:, 0], self.data[:, 1], self.data[:, 2], ".", alpha=0.5)


if __name__ == "__main__":
    gmmdata = GMMData(dim=2, sample_size=1000, random_seed=42)
    dataset = gmmdata.get_data()
    fig = plt.figure()
    ax = plt.axes()  # projection='3d' is required if you use 3-D data.
    gmmdata.visualize(ax)
    plt.show()
