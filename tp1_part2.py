import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.data

y = digits.target

y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1

X_train , X_test , y_train , y_test = train_test_split(X, y_one_hot , test_size=0.3, random_state=42)

X_test , X_validation , y_test , y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1])) # weights of shape KxL

best_W = None
best_accuracy = 0
lr = 0.001
nb_epochs = 50
minibatch_size = len(y) // 20

losses = []
accuracies = []

def softmax(x):
	e = np.exp(x - np.max(x, axis=0))
	return e / e.sum(axis = 0)

def get_accuracy(X, y, W):
	prob = softmax(W.dot(X.T))
	_y = np.zeros((X.shape[0], W.shape[0]))
	_y[np.arange(_y.shape[0]), prob.argmax(axis=0)] = 1
	return np.mean((y * _y).sum(axis=1))

def get_grads(y, y_pred, X):
	grad = -(y-y_pred).T.dot(X)
	return grad

def get_loss(y, y_pred):
	return -np.mean((y*y_pred).sum(axis=1))

for epoch in range(nb_epochs):
	loss = 0
	accuracy = 0
	random_idxs = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
	X_shuffled = X_train[random_idxs,:]
	y_shuffled = y_train[random_idxs,:]
	for i in range(0, X_train.shape[0], minibatch_size):
		X_batch = X_shuffled[i:i+minibatch_size,:]
		y_batch = y_shuffled[i:i+minibatch_size,:]
		y_pred = softmax(W.dot(X_batch.T)).T
		grad = get_grads(y_batch, y_pred, X_batch)
		W -= lr*grad
	loss = get_loss(y_train, softmax(W.dot(X_train.T)).T)
	losses.append(loss) # compute the loss on the train set
	accuracy = get_accuracy(X_validation, y_validation, W)
	accuracies.append(accuracy) # compute the accuracy on the validation set
	if accuracy > best_accuracy:
		best_W = W
		best_accuracy = accuracy

accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
print(accuracy_on_unseen_data)

plt.plot(losses)
plt.figure()

plt.imshow(best_W [4,:].reshape(8 ,8))
plt.show()
