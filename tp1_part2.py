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

X_test , X_validation , y_test , y_validation = train_test_split(X_test, y_test, t est_size=0.5, random_state=42)

W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1])) # weights of shape KxL

best_W = None
best_accuracy = 0
lr = 0.001
nb_epochs = 50
minibatch_size = len(y) // 20

losses = []
accuracies = []

def softmax(x):
	
