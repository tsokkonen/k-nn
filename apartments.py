import numpy as np
import learner as lrn

# Prices (euro/m^2), sizes (m^2) and age (years in 2017) of 3498
# homes sold in Helsinki in 2017
xs = np.loadtxt('features.txt')

# Labels to indicate these were apartments (1) or other kinds (0) of homes
ys = np.loadtxt('labels.txt').astype('int')

# Split the data to training and test sets (n is the size of training set)
n = 3000
xs_train = xs[:n]
ys_train = ys[:n]
xs_test  = xs[n:]
ys_test  = ys[n:]

# Predict labels for apartments in the test set using 1-NN rule
pred1 = []
for x in xs_test:
    y = lrn.one_nn(x, xs_train, ys_train)
    pred1.append(y)

ys_pred1 = np.array(pred1)

# Compute the predictor's loss
loss1 = (ys_pred1 != ys_test).sum() / float(ys_test.size)

print ('Predictions based on 1-NN rule fail in %s percent of the '
       'cases') % round(100*loss1, 2)

# Predict labels for apartments in the test set using 3-NN rule
pred3 = []
for x in xs_test:
    y = lrn.k_nn(3, x, xs_train, ys_train)
    pred3.append(y)

ys_pred3 = np.array(pred3)

loss3 = (ys_pred3 != ys_test).sum() / float(ys_test.size)

print ('Predictions based on 3-NN rule fail in %s percent of the '
       'cases') % round(100*loss3, 2)
