import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression import *

digits = datasets.load_digits()

X = digits.data

y = digits.target

y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1

lrs = [0.1, 0.01, 0.001, 0.0001]
nb_epochs = 50
minibatch_sizes = [1, 20, 200, 1000]


for minibatch_size in minibatch_sizes:
	for lr in lrs:
		W, log_test, log_validation, accuracies = get_params(X, y_one_hot, nb_epochs, lr, minibatch_size)

		plt.clf()
		plt.plot(log_test, 'b', label = 'test')
		plt.plot(log_validation, 'r', label = 'validation')
		plt.xlabel('Epoch')
		plt.ylabel('Average negative log likelihood')
		plt.legend(loc='best')
		plt.title('lr = ' + str(lr) + ', minibatch_size = ' + str(minibatch_size))
		plt.savefig("img/lr=" + str(lr) + "_minibatch=" + str(minibatch_size) + ".png")
