import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class fadata:
    def __init__(self, sample_size=100, n=10, m=3, sqrsigma=0.1, random_seed=None):
        self.sample_size = sample_size
        self.n = n
        self.m = m
        self.sqrsigma = sqrsigma
        # default mu = 0
        if random_seed is not None:
            np.random.seed(random_seed)
        self.genA()

    def genA(self):
        # A nxm
        self.A = np.random.normal(size=(self.n,self.m))

    def data(self):
        self.X = []
        for i in range(self.sample_size):
            # y = 0, I_{mx1}
            y = np.random.normal(size=(self.m))
            # e = 0, sqrsigma I_{nx1}
            e = np.random.normal(scale=np.sqrt(self.sqrsigma),size=(self.n))
            # x = Ay + mu(0) + e
            x = self.A @ y + e
            self.X.append(x)
        return self.X

    def visualize(self, ax):
        data = np.array(self.X)
        if self.n == 3:
            ax.plot3D(data[:, 0], data[:, 1], data[:, 2], ".", alpha=0.5)
        elif self.n == 2:
            ax.plot(data[:, 0], data[:, 1], ".", alpha=0.5)

if __name__ == "__main__":
    fadata = fadata(sample_size=1000,n=3,m=3)
    X = fadata.data()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax = plt.axes()
    fadata.visualize(ax)
    plt.show()
