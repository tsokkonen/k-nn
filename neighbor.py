import numpy as np

class Neighbor:
    def __init__(self, k, loss=None):
        self.k = k
        self.loss = loss
        
    def dist(self, x, xs):
        diffs = xs - x
        f = np.linalg.norm
        return np.apply_along_axis(f, 1, diffs)

    def predict(self, x, xs, ys):
        ds = self.dist(x, xs)
        dtype = [('distance', float), ('label', int)]
        a = np.array(zip(ds, ys), dtype=dtype)
        b = np.sort(a, order='distance')
        return b['label'][:self.k].mean().round().astype('int')

    def test(self, xs_train, xs_test, ys_train, ys_test):
        predictions = []
        for x in xs_test:
            y = self.predict(x, xs_train, ys_train)
            predictions.append(y)
        ys_pred = np.array(predictions)
        self.loss = (ys_pred != ys_test).sum() / float(ys_test.size)
    
    def __repr__(self):
        return '[Rule: %s-NN, Loss: %s]' % (self.k, self.loss)

if __name__ == '__main__':
    from neighbor import Neighbor
    xs = np.loadtxt('features.txt')
    ys = np.loadtxt('labels.txt')

    n = 3000
    xs_train = xs[:n]
    ys_train = ys[:n]
    xs_test  = xs[n:]
    ys_test  = ys[n:]

    nn1 = Neighbor(1)
    nn3 = Neighbor(3)
    nn5 = Neighbor(5)
    nn7 = Neighbor(7)
    rules = (nn1, nn3, nn5, nn7)

    x = np.array([54.5, 4000, 90])
    for r in rules: print r, r.predict(x, xs_train, ys_train)

    for r in rules:
        r.test(xs_train, xs_test, ys_train, ys_test)
        print r
