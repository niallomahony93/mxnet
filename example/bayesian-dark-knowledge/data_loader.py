import numpy


def load_mnist(training_num=50000):
    dat = numpy.load('mnist.npz')
    X = (dat['X'][:training_num] / 126.0).astype('float32')
    Y = dat['Y'][:training_num]
    X_test = (dat['Xtest'] / 126.0).astype('float32')
    Y_test = dat['Ytest']
    Y = Y.reshape((Y.shape[0], ))
    Y_test = Y_test.reshape((Y_test.shape[0], ))
    return X, Y, X_test, Y_test


def load_toy():
    training_data = numpy.loadtxt('toy_data_train.txt')
    testing_data = numpy.loadtxt('toy_data_test_whole.txt')
    X = training_data[:, 0].reshape((training_data.shape[0], 1))
    Y = training_data[:, 1].reshape((training_data.shape[0], 1))
    X_test = testing_data[:, 0].reshape((testing_data.shape[0], 1))
    Y_test = testing_data[:, 1].reshape((testing_data.shape[0], 1))
    return X, Y, X_test, Y_test

def load_synthetic(theta1, theta2, sigmax, num=20):
    flag = numpy.random.randint(0, 2, (num,))
    X = flag * numpy.random.normal(theta1, sigmax, (num, )) \
                    + (1.0 - flag) * numpy.random.normal(theta1 + theta2, sigmax, (num, ))
    return X