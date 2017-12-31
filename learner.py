import numpy as np
import matplotlib.pyplot as plt

def dist(a, b):
    return np.linalg.norm(a - b)

def read_data(fname):
    return np.genfromtxt(fname, delimiter=",")
    
def compute_distances(xs):
    dists = []
    for i in range(len(xs)):
        for x in xs:
            dists.append(dist(xs[i], x))
    return np.reshape(dists, (len(xs), len(xs)))

def show_samples(xs, ys):
    plt.scatter(xs[:,0], xs[:,1], c=ys, cmap='Greys')
    return plt.show()
