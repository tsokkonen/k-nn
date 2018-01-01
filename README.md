# k-nn
Implementation of 1-NN learning rule for the *k* nearest neighbors algorithm.

# Example usage
To load the training and label sets, plot the sample points, and classify one new point, do the following:
```Python
>>> import numpy as np
>>> import learner as lrn
>>> xs = lrn.read_data("xs.txt")
>>> ys = lrn.read_data("ys.txt")
>>> lrn.show_samples(xs, ys)
>>> x = np.array([9, 1])
>>> lrn.one_nn(x, xs, ys)
1.0
```
