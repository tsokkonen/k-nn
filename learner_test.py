import numpy as np
import unittest
import learner

class TestLearnerFunctions(unittest.TestCase):

    def test_dist(self):
        x = np.array([2, 2])
        xs = learner.read_data("xs.txt")
        result = learner.dist(x, xs)
        expected = 4.1231
        diff = result[1] - expected
        self.assertTrue(abs(diff) < 0.0001)

    def test_read_data(self):
        xs = learner.read_data("xs.txt")
        self.assertEquals(xs.shape, (8, 2))

    def test_one_nn(self):
        xs = learner.read_data("xs.txt")
        ys = learner.read_data("ys.txt")
        x = np.array([9, 9])
        result = learner.one_nn(x, xs, ys)
        expected = 1.0
        diff = result - expected
        self.assertTrue(abs(diff) < 0.0001)

if __name__ == '__main__':
    unittest.main()
