import numpy as np
import matplotlib.pyplot as plt


class GaussianMixtureModel(object):

    def __init__(self):
        self.m = None  # shape: (n_clusters, data_dimension)
        self.class_distribution = None  # shape: (n_data, n_clusters)

    def fit(self, data, n_clusters=3):
        self.m = self._initialize_m(data, n_clusters)
        threshold = 0.00001
        i = 0
        while True:
            print(i)
            print(self.m)
            self.class_distribution = self._e_step(data, self.m, n_clusters)
            self.draw_figure_2d(data, step=i, n_clusters=n_clusters)
            m = self._m_step(data, self.class_distribution, n_clusters)
            if ((m - self.m) ** 2).sum() < threshold:  # 終了条件
                self.m = m
                break
            self.m = m
            i += 1

    @staticmethod
    def _initialize_m(data, n_clusters):
        m = np.random.rand(n_clusters, data.shape[1]) * (data.max(axis=0) - data.min(axis=0)) + data.min(axis=0)
        return m

    @classmethod
    def _e_step(cls, data, m, n_clusters):
        class_distribution = np.zeros((data.shape[0], n_clusters))
        for i, one_data in enumerate(data):
            tmp = []
            for c in range(n_clusters):
                prob = cls._gauss(one_data, m[c], sigma=0.08)
                # sigma = 0.1が最適っぽい
                # sigmaが大きすぎると全体の点の分布に引っ張られてしまい全てのmが中央に集まってしまう
                # sigmaが小さすぎると発散してしまう
                class_distribution[i, c] = prob
                tmp.append(prob)
            class_distribution[i] /= np.array(tmp).sum()
        return class_distribution

    @classmethod
    def _m_step(cls, data, class_distribution, n_clusters):
        m = np.zeros((n_clusters, data.shape[1]))
        for c in range(n_clusters):
            numer = []
            denom = []
            for i, data_one in enumerate(data):
                numer.append(class_distribution[i, c] * data_one)
                denom.append(class_distribution[i, c])
            m[c] = np.array(numer).sum(axis=0) / np.array(denom).sum()
        return m

    @staticmethod
    def _gauss(x, mu, sigma):
        return np.exp(- ((x - mu) ** 2).sum() / (2 * sigma ** 2))

    def draw_figure_2d(self, data, step, n_clusters):
        if self.class_distribution is None:
            raise ValueError('please fit first!')

        data_class = self.class_distribution.argmax(axis=1)

        for data_class_one, c in zip(range(n_clusters), ['c', 'b', 'g', 'y', 'k', 'r']):  # この色のところなんとかしたい
            # if data.shape[1] != 2:
            #     raise ValueError('data should be 2d')
            x = data[data_class == data_class_one][:, 0]
            y = data[data_class == data_class_one][:, 1]
            plt.scatter(x, y, c=c)
            plt.scatter(self.m[data_class_one, 0], self.m[data_class_one, 1], c=c, marker='x')
        plt.savefig(f'figures/gaussian_mixture_model_{step}.png')
        plt.clf()

