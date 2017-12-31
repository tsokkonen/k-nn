import numpy as np
import unittest
import learner

class TestLearnerFunctions(unittest.TestCase):
    def setUp(self):
        self.xs = learner.read_data("xs.txt")
        self.ys = learner.read_data("ys.txt")

    def tearDown(self):
        self.xs.dispose()
        self.ys.dispose()
        self.xs = None
        self.ys = None

    def test_dist(self):
        x = np.array([2, 2])
        result = learner.dist(x, self.xs)
        expected = 4.1231
        diff = result[1] - expected
        self.assertTrue(abs(diff) < 0.0001)

    def test_read_data(self):
        self.assertEquals(self.xs.shape, (8, 2))

    def test_one_nn(self):
        x = np.array([9, 9])
        result = learner.one_nn(x, self.xs, self.ys)
        expected = 1.0
        diff = result - expected
        self.assertTrue(abs(diff) < 0.0001)

if __name__ == '__main__':
    unittest.main()
