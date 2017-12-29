import numpy as np
import unittest
import bin

from io import BytesIO

class TestBinFunctions(unittest.TestCase):

    def test_dist(self):
        x1 = np.array([2, 2])
        x2 = np.array([3, 6])
        result = bin.dist(x1, x2)
        expected = 4.1231
        diff = result - expected
        self.assertTrue(abs(diff) < 0.0001)

    def test_read_data(self):
        xs = bin.read_data("xs.txt")
        self.assertEquals(xs.shape, (8, 2))

    def test_compute_distances(self):
        xs = bin.read_data("xs.txt")
        dists = bin.compute_distances(xs)
        self.assertEquals(dists.shape, (8, 8))

if __name__ == '__main__':
    unittest.main()
