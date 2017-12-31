import numpy as np
import matplotlib.pyplot as plt

def dist(x, xs):
       diffs = xs - x
       f = np.linalg.norm
       return np.apply_along_axis(f, 1, diffs)

def one_nn(x, xs, ys):
       d = dist(x, xs)
       return ys[d.argmin()]

def read_data(fname):
    return np.genfromtxt(fname, delimiter=",")
    
def show_samples(xs, ys):
    plt.scatter(xs[:,0], xs[:,1], c=ys, cmap='Greys')
    return plt.show()
