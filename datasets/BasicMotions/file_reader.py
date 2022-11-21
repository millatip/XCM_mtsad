import numpy as np
train_X = np.load('X_train.npy')
train_Y = np.load('y_train.npy')

print(train_X.shape)
print(len(train_X))
print(train_X[1].shape)
print(train_Y.shape)
