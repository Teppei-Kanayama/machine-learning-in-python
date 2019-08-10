from itertools import combinations, product

import numpy as np
import matplotlib.pyplot as plt

# unittestで良い感じに検証できるようにする
# sample_data = np.array([[0, 0],
#                         [1, 1],
#                         [1.1, 1],
#                         [1, 1.1],
#                         [2, 2]])

sample_data = np.random.rand(100, 2)


class AgglomerativeClustering(object):
    def __init__(self):
        self._clustered_data = None

    def fit(self, data: np.ndarray, n_class: int = 2, method='centroid'):
        data_list = [[i] for i in data]
        while True:
            max_distance_pair = self._get_max_distance_pair(data_list, method)
            data_list[max_distance_pair[0]] = data_list[max_distance_pair[0]] + data_list[max_distance_pair[1]]
            del data_list[max_distance_pair[1]]

            if len(data_list) <= n_class:
                break
        self._clustered_data = list(map(lambda x: np.array(x), data_list))

    @classmethod
    def _get_max_distance_pair(cls, data_list, method):
        min_distance = 10000000
        min_distance_pair = (0, 1)
        for i, j in combinations(range(len(data_list)), 2):
            distance = cls._measure_distance(data_list[i], data_list[j], method)
            if distance < min_distance:
                min_distance_pair = (i, j)
                min_distance = distance
        return min_distance_pair

    def get_clustered_data(self):
        if not self._clustered_data:
            raise ValueError('please fit first!')

        return self._clustered_data

    def draw_figure_2d(self):
        if not self._clustered_data:
            raise ValueError('please fit first!')

        for data, c in zip(self._clustered_data, ['c', 'b', 'g', 'y', 'k', 'r', 'm']):  # この色のところなんとかしたい
            if data.shape[1] != 2:
                raise ValueError('data should be 2d')

            plt.scatter(data[:, 0], data[:, 1], c=c)

        plt.savefig('figures/agglomerative_clustering.png')

    @staticmethod
    def _measure_distance(data_list1, data_list2, method='complete_link'):

        dist = lambda x, y: np.sqrt(((x - y) ** 2).sum())

        if method == 'centroid':
            centor1 = np.array(data_list1).mean(axis=0)
            centor2 = np.array(data_list2).mean(axis=0)
            distance = dist(centor1, centor2)
        elif method == 'single_link':
            # なんかこれfor文回さなくてもうまくかける気がする
            min_distance = 10000000
            for x, y in product(data_list1, data_list2):
                tmp_distance = dist(x, y)
                if tmp_distance < min_distance:
                    min_distance = tmp_distance
            distance = min_distance
        elif method == 'complete_link':
            # なんかこれfor文回さなくてもうまくかける気がする
            max_distance = 0
            for x, y in product(data_list1, data_list2):
                tmp_distance = dist(x, y)
                if tmp_distance > max_distance:
                    max_distance = tmp_distance
            distance = max_distance
        else:
            raise NotImplementedError
        return distance


if __name__ == '__main__':
    agglomerative_clustering = AgglomerativeClustering()
    agglomerative_clustering.fit(sample_data, n_class=3, method='complete_link')
    agglomerative_clustering.draw_figure_2d()
