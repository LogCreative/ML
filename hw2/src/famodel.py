from datagen import fadata
from sklearn.decomposition import FactorAnalysis
import numpy as np

class famodel:
    def __init__(self, dataset, max_m = 5, max_iter=1000, rand_seed=None):
        self.dataset = dataset
        self.max_m = max_m
        self.sample_size = len(dataset)
        self.n = len(dataset[0])
        self.max_iter = max_iter
        if rand_seed is not None:
            np.random.seed(rand_seed)

    def train(self):
        self.max_aic_m = 0
        max_aic = - np.inf
        self.max_bic_m = 0
        max_bic = - np.inf
        for m in range(1, self.max_m + 1):
            model = FactorAnalysis(m, max_iter=self.max_iter)
            model.fit(self.dataset)
            # score is the AVERAGE of log-like
            loglike = model.score(self.dataset) * self.sample_size
            num_free = self.n * m + 1 - m * (m - 1) / 2
            aic = loglike - num_free
            bic = loglike - num_free * np.log(self.sample_size) / 2
            if aic > max_aic:
                self.max_aic_m = m
                max_aic = aic
            if bic > max_bic:
                self.max_bic_m = m
                max_bic = bic

if __name__ == '__main__':

    random_seeds = [42, 57, 89, 110, 211, 985, 1314, 2567, 3388, 10000]

    with open('nm.dat','w') as f:
        dataline = 'n\tm\taics\tbics\taicacc\tbicacc'
        print(dataline)
        f.write(dataline + '\n')
        for n in [5,10,15,20,25]:
            for m in [1,2,3,4,5]:
                if m <= n:
                    aic_acc = 0
                    aic_score = 0
                    bic_acc = 0
                    bic_score = 0
                    for rand_seed in random_seeds:
                        model = famodel(fadata(sample_size=200, sqrsigma=0.1, n=n, m=m, random_seed=rand_seed).data(), rand_seed=rand_seed)
                        model.train()
                        aic_acc += 1 if model.max_aic_m == m else 0
                        aic_score += (model.max_aic_m - m) / m
                        bic_acc += 1 if model.max_bic_m == m else 0
                        bic_score += (model.max_bic_m - m) / m
                    aic_acc /= len(random_seeds)
                    aic_score /= len(random_seeds)
                    bic_acc /= len(random_seeds)
                    bic_score /= len(random_seeds)
                    dataline = '{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(n, m, aic_score, bic_score, aic_acc, bic_acc)
                    print(dataline)
                    f.write(dataline + '\n')

    with open('Ns.dat','w') as f:
        dataline = 'N\ts\taics\tbics\taicacc\tbicacc'
        print(dataline)
        f.write(dataline + '\n')
        fixed_n = 10
        target_m = 5
        for N in [20, 50, 100, 200, 500, 1000]:
            for sig in [0, 0.05, 0.1, 0.2, 0.5, 1, 2]:
                aic_acc = 0
                aic_score = 0
                bic_acc = 0
                bic_score = 0
                for rand_seed in random_seeds:
                    model = famodel(fadata(sample_size=N, sqrsigma=sig,n=fixed_n, m=target_m, random_seed=rand_seed).data(), rand_seed=rand_seed)
                    model.train()
                    aic_acc += 1 if model.max_aic_m == target_m else 0
                    aic_score += (model.max_aic_m - target_m) / target_m
                    bic_acc += 1 if model.max_bic_m == target_m else 0
                    bic_score += (model.max_bic_m - target_m) / target_m
                aic_acc /= len(random_seeds)
                aic_score /= len(random_seeds)
                bic_acc /= len(random_seeds)
                bic_score /= len(random_seeds)
                dataline = '{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(N, sig, aic_score, bic_score, aic_acc, bic_acc)
                print(dataline)
                f.write(dataline + '\n')
