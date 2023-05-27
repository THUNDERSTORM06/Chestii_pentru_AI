import numpy as np
from sklearn.metrics import mean_squared_error


def univariate_regression(train_inputs, test_inputs, train_outputs, test_outputs, learning_rate, epochs):
    m = 124
    x = np.c_[np.ones((124, 1)), train_inputs]
    x_transpose = x.transpose()
    beta = np.array([0, 0])

    for _ in range(epochs):
        ipoteza = np.dot(x, beta)
        loss = ipoteza - train_outputs
        gradient = np.dot(x_transpose, loss) / m
        beta = beta - learning_rate * gradient

    computed_outputs = [beta[0] + test_inputs[i] * beta[1] for i in range(len(test_inputs))]
    print('Outputs: ', test_outputs)
    print('Computed outputs: ', computed_outputs)

    error = mean_squared_error(test_outputs, computed_outputs)
    print('Error: ', error)


def multivariate_regression(trainInputs, trainOutputs, alpha, iters):
    happiness_new = np.reshape(trainOutputs, (len(trainOutputs), 1))
    gdp = np.c_[np.ones((len(trainInputs), 1)), trainInputs]
    theta = np.random.randn(len(gdp[0]), 1) # Generate an initial value of vector θ from the original independent
    # variables matrix
    for _ in range(iters):
        gradients = 2 / len(happiness_new) * np.dot(np.transpose(gdp), (np.dot(gdp, theta) - happiness_new))
        theta = theta - alpha * gradients
    model = f"the learnt model: f(x1, x2) = {theta[0][0]} + {theta[1][0]} * x1 + {theta[2][0]} * x2"
    return model
