from sklearn import cluster
from datagen import GMMData
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy


class GMM:
    def __init__(self, dataset, max_k=5, max_iter=10000):
        self.dataset = dataset
        self.max_k = max_k
        self.max_iter = max_iter

        self.aic_crts = []
        self.bic_crts = []
        self.best_k_aic = 0
        self.best_k_bic = 0

    def select(self):
        models = []
        for k in range(self.max_k):
            model = GaussianMixture(n_components=k + 1, max_iter=self.max_iter)
            model.fit(self.dataset)
            models.append(copy.deepcopy(model))
            self.aic_crts.append(model.aic(self.dataset))
            self.bic_crts.append(model.bic(self.dataset))
        self.best_k_aic = np.argmin(self.aic_crts)
        self.best_k_aic_model_ = models[self.best_k_aic]
        self.best_k_aic += 1
        self.best_k_bic = np.argmin(self.bic_crts)
        self.best_k_bic_model_ = models[self.best_k_bic]
        self.best_k_bic += 1

    def visualize(self, ax_aic, ax_bic):
        labels_aic = self.best_k_aic_model_.predict(self.dataset)
        ax_aic.set_title("AIC K=%d" % self.best_k_aic)
        ax_aic.scatter(self.dataset[:, 0], self.dataset[:, 1], c=labels_aic)
        labels_bic = self.best_k_bic_model_.predict(self.dataset)
        ax_bic.set_title("BIC K=%d" % self.best_k_bic)
        ax_bic.scatter(self.dataset[:, 0], self.dataset[:, 1], c=labels_bic)

class VBEMGMM:
    def __init__(
        self,
        dataset,
        max_k = 5,
        max_iter = 10000
    ):
        self.dataset = dataset
        self.model = BayesianGaussianMixture(n_components=max_k, max_iter=max_iter)
        self.best_k_vb = 0
        self.model.fit(self.dataset)

    def visualize(self, ax):
        labels = self.model.predict(self.dataset)
        self.best_k_vb = len(np.unique(labels))
        ax.scatter(self.dataset[:, 0], self.dataset[:, 1], c=labels)
        ax.set_title("VB K=%d" % self.best_k_vb)

class GMMTest:
    def __init__(self, cluster_nums, sample_sizes, random_seeds):
        self.cluster_nums = cluster_nums
        self.sample_sizes = sample_sizes
        self.random_seeds = random_seeds
        self.logger_str = ""
        self.reset_counter()
        self.result = []

    def reset_counter(self):
        self.aic_correct = 0
        self.bic_correct = 0
        self.vb_correct = 0

    def test(self):
        plt.ion()
        print("#case_num=%2d" % len(self.random_seeds))
        print("K\tN\tAIC\tBIC\tVB")
        for cluster_num in self.cluster_nums:
            for sample_size in self.sample_sizes:
                self.reset_counter()
                for random_seed in self.random_seeds:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                    fig.suptitle("target=%d" % cluster_num)
                    dataset = GMMData(
                        cluster_num=cluster_num, sample_size=sample_size, random_seed=random_seed
                    ).get_data()
                    gmmsel = GMM(dataset, cluster_num + 3)
                    gmmsel.select()
                    gmmsel.visualize(ax1, ax2)
                    vbgmm = VBEMGMM(dataset, cluster_num + 3)
                    vbgmm.visualize(ax3)
                    filename = "em_%d_%d_%d.png" % (cluster_num, sample_size, random_seed)
                    if gmmsel.best_k_aic != cluster_num or gmmsel.best_k_bic != cluster_num or vbgmm.best_k_vb != cluster_num:
                        plt.savefig("res/%s" % filename)
                        plt.pause(0.5)
                        plt.clf()
                        plt.close('all')

                    if gmmsel.best_k_aic == cluster_num:
                        self.aic_correct += 1
                    else:
                        self.logger_str += "AIC Failed @ %s: get K=%d, expected K=%d\n" % (filename, gmmsel.best_k_aic, cluster_num)
                    
                    if gmmsel.best_k_bic == cluster_num:
                        self.bic_correct += 1
                    else:
                        self.logger_str += "BIC Failed @ %s: get K=%d, expected K=%d\n" % (filename, gmmsel.best_k_bic, cluster_num)

                    if vbgmm.best_k_vb == cluster_num:
                        self.vb_correct += 1
                    else:
                        self.logger_str += "VB Failed @ %s: get K=%d, expected K=%d\n" % (filename, vbgmm.best_k_vb, cluster_num)
                    
                res = [cluster_num, sample_size, self.aic_correct, self.bic_correct, self.vb_correct]
                print("%2d\t%3d\t%2d\t%2d\t%2d" % tuple(res))
                self.result.append(res)
        print("\nLogger:")
        print(self.logger_str)
        plt.ioff()

    def visualize(self, ax):
        result = np.array(self.result)
        X = np.reshape(result[:,0], (len(self.cluster_nums), len(self.sample_sizes)))
        Y = np.reshape(result[:,1], (len(self.cluster_nums), len(self.sample_sizes)))
        Z_AIC = np.reshape(result[:,2], (len(self.cluster_nums), len(self.sample_sizes)))
        Z_BIC = np.reshape(result[:,3], (len(self.cluster_nums), len(self.sample_sizes)))
        Z_VB = np.reshape(result[:,4], (len(self.cluster_nums), len(self.sample_sizes)))
        ax.plot_surface(X, Y, Z_AIC, label='AIC', cmap="Reds")
        ax.plot_surface(X, Y, Z_BIC, label='BIC', cmap="Blues")
        ax.plot_surface(X, Y, Z_VB, label='VB', cmap="Greens")
        ax.set_xlabel('cluster_num')
        ax.set_ylabel('sample_size')
        ax.set_zlabel('pass_num')

if __name__ == "__main__":
    random_seeds = [42, 56, 71, 101, 141, 201, 231, 271, 301, 401, 451]
    cluster_nums = [3, 4, 5, 6, 7, 8]
    sample_sizes = [50, 100, 200, 400, 800]
    tester = GMMTest(cluster_nums, sample_sizes, random_seeds)
    tester.test()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    tester.visualize(ax)
    plt.savefig('result.png')
    plt.show()