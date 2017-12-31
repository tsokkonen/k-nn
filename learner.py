import numpy as np

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
