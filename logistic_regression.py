import numpy as np
from sklearn.model_selection import train_test_split


def softmax(x):
	# To make sure that the function is numerically
	# stable we substract the maximum before applying
	# the exponential function.
	e = np.exp(x - np.max(x, axis=0))
	return e / np.sum(e, axis=0)

def get_accuracy(X, y, W):
	# Compute the probabilities using W.
	prob = softmax(W.dot(X.T))
	# We look for labels with maximum probability.
	labels = np.zeros((X.shape[0], W.shape[0]))
	labels[np.arange(labels.shape[0]), prob.argmax(axis=0)] = 1
	return np.mean(np.sum(y*labels, axis=1))

def get_grads(y, y_pred, X):
	# We use the expression of the gradiant
	grad = -(y-y_pred).T.dot(X) / len(y)
	return grad

def get_loss(y, y_pred):
	# We use the expression of the loss function
	return -np.mean(np.sum(y*y_pred, axis=1))

def get_log_likelihood(X, y, W):
	# We compute W . X
	z = (W.dot(X.T))
	# We substract the maximum for the numerical stability.
	z = z - np.max(z, axis=0)
	return -np.mean(np.sum(y.T*z, axis=0) - np.log(np.sum(np.exp(z), axis=0)))

def get_params(X, y, nb_epochs, lr, minibatch_size, test_size=0.3, validation_size=0.5):

    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_test , X_validation , y_test , y_validation = train_test_split(X_test, y_test, test_size=validation_size, random_state=42)

    W = np.random.normal(0, 0.01, (y.shape[1], X.shape[1]))

    best_W = None
    best_accuracy = 0
    losses = []
    accuracies = []
    log_validation = [] # the negative log vraisemblance on the validation set.
    log_test = [] # the negative log vraisemblance on the test set.
    for epoch in range(nb_epochs):
        loss = 0
        accuracy = 0
        # Shuffle the train set
        random_idxs = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
        X_shuffled = X_train[random_idxs,:]
        y_shuffled = y_train[random_idxs,:]
        for i in range(0, X_train.shape[0], minibatch_size):
            X_batch = X_shuffled[i:i+minibatch_size,:]
            y_batch = y_shuffled[i:i+minibatch_size,:]
            # Compute the probabilities for the current batch set.
            y_batch_pred = softmax(W.dot(X_batch.T)).T
            # Use this probabilities to compute the gradiant
            grad = get_grads(y_batch, y_batch_pred, X_batch)
            # Make a descent and multiply the gradiant by the learning rate.
            W -= lr*grad

		# Compute the predictions probabilities on the train set
        y_train_pred = softmax(W.dot(X_train.T)).T
        # Compute the loss on the train set
        loss = get_loss(y_train, y_train_pred)
        losses.append(loss) # compute the loss on the train set

        log_test.append(get_log_likelihood(X_test, y_test, W))
        log_validation.append(get_log_likelihood(X_validation, y_validation, W))
        accuracy = get_accuracy(X_validation, y_validation, W)
        accuracies.append(accuracy) # compute the accuracy on the validation set
		# select the best parameters based on the validation accuracy
        if accuracy > best_accuracy:
            best_W = np.copy(W)
            best_accuracy = accuracy

    return best_W, log_test, log_validation, accuracies
