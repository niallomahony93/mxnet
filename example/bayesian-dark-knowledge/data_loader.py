import numpy


def load_mnist():
    dat = numpy.load('mnist.npz')
    X = dat['X'][:50000] / 126.0
    Y = dat['Y'][:50000]
    X_test = dat['Xtest'] / 126.0
    Y_test = dat['Ytest']
    Y = Y.reshape((Y.shape[0], ))
    Y_test = Y_test.reshape((Y_test.shape[0], ))
    return X, Y, X_test, Y_test


def load_toy():
    training_data = numpy.loadtxt('toy_data_train.txt')
    testing_data = numpy.loadtxt('toy_data_test.txt')
    X = training_data[:, 0].reshape((training_data.shape[0], 1))
    Y = training_data[:, 1].reshape((training_data.shape[0],))
    X_test = testing_data[:, 0].reshape((testing_data.shape[0], 1))
    Y_test = testing_data[:, 1].reshape((testing_data.shape[0],))
    return X, Y, X_test, Y_test


