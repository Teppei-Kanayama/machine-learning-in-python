import unittest
import numpy as np

from clustering.gaussian_mixture_model import GaussianMixtureModel


class TestGaussianMixtureModel(unittest.TestCase):

    def setUp(self):
        self.sample_data1 = np.array([[-1, 10],
                                      [1, 5],
                                      [1.1, 3]])

        self.sample_data2 = np.array([[0, 0],
                                    [1, 1],
                                    [1.1, 1],
                                    [1, 1.1],
                                    [2, 2]])

        self.sample_data3 = np.random.rand(100, 2)

        self.m = np.array([[1, 2],
                           [3, 4]])

        self.class_probability = np.array([[0.1, 0.9],
                                        [0.9, 0.1],
                                        [0.4, 0.6]])

    def test_fit(self):
        gaussian_mixture_model = GaussianMixtureModel()
        gaussian_mixture_model.fit(self.sample_data3, n_clusters=3)
    #
    # def test_initialize_m(self):
    #     n_clusters = 2
    #     m = GaussianMixtureModel._initialize_m(self.sample_data1, n_clusters=n_clusters)
    #     self.assertEqual(m.shape[0], n_clusters)
    #     self.assertEqual(m.shape[1], self.sample_data1.shape[1])
    #     self.assertTrue(m[:, 0].max() <= 1.1)
    #     self.assertTrue(m[:, 0].min() >= -1)
    #     self.assertTrue(m[:, 1].max() <= 10)
    #     self.assertTrue(m[:, 1].min() >= 3)
    #
    # def test_e_step(self):
    #     output = GaussianMixtureModel._e_step(self.sample_data1, self.m, n_clusters=2)
    #     np.testing.assert_array_equal(output.sum(axis=1), np.ones(self.sample_data1.shape[0]))
    #
    # def test_m_step(self):
    #     output = GaussianMixtureModel._m_step(self.sample_data1, self.class_probability, n_clusters=2)
    #     self.assertEqual(output.shape, (2, 2))
    #     self.assertEqual(output[0, 0], (-1 * 0.1 + 1 * 0.9 + 1.1 * 0.4) / (0.1 + 0.9 + 0.4))
    #     self.assertEqual(output[0, 1], (10 * 0.1 + 5 * 0.9 + 3 * 0.4) / (0.1 + 0.9 + 0.4))
    #     self.assertEqual(output[1, 0], (-1 * 0.9 + 1 * 0.1 + 1.1 * 0.6) / (0.9 + 0.1 + 0.6))
    #     self.assertEqual(output[1, 1], (10 * 0.9 + 5 * 0.1 + 3 * 0.6) / (0.9 + 0.1 + 0.6))

