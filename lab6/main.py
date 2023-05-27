import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from developed_code.my_regression import *
from reader.reader import *
from sklearn_tool import regression
from profa import main2
import matplotlib.pyplot as plt
import numpy as np



def loadData(fileName, inputVariabName, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    variabilaSelectata = dataNames.index(inputVariabName)
    inputs = [float(data[i][variabilaSelectata]) for i in range(len(data))]
    outputSelectat = dataNames.index(outputVariabName)
    outputs = [float(data[i][outputSelectat]) for i in range(len(data))]

    return inputs, outputs


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotData(x1, y1, x2=None, y2=None, x3=None, y3=None, title=None):
    plt.plot(x1, y1, 'ro', label='train data')
    if x2:
        plt.plot(x2, y2, 'b-', label='learnt model')
    if x3:
        plt.plot(x3, y3, 'g^', label='test data')
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'v4_world-happiness-report-2017.csv')
    inputs, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
    gdp, outputs, freedom = reader(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')

    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    # training step
    xx = [[el] for el in trainInputs]
    regressor = linear_model.LinearRegression()
    # regressor = linear_model.SGDRegressor(max_iter =  10000)
    regressor.fit(xx, trainOutputs)
    w0, w1 = regressor.intercept_, regressor.coef_

    # plot the model
    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]
    regression.function(gdp, freedom, outputs)

    # learnt model
    my_linear_regression(gdp, freedom, outputs)

    computedValidationOutputs = regressor.predict([[x] for x in validationInputs])

    # plotDataHistogram(inputs, 'capita GDP')
    # plotDataHistogram(outputs, 'Happiness score')
    plotData(inputs, outputs, [], [], [], [], 'capita vs. hapiness')
    plotData(trainInputs, trainOutputs, [], [], validationInputs, validationOutputs, "train and test data")
    plotData([], [], validationInputs, computedValidationOutputs, validationInputs, validationOutputs,
             "predictions vs real test data")
    plotData(trainInputs, trainOutputs, xref, yref, [], [], title="train data and model")

    # compute the differences between the predictions and real outputs
    error = 0.0

    for t1, t2 in zip(computedValidationOutputs, validationOutputs):
        error += (t1 - t2) ** 2

    error = error / len(validationOutputs)
    print("prediction error (manual): ", error)

    error = mean_squared_error(validationOutputs, computedValidationOutputs)
    print("prediction error (tool): ", error)


main()
#main2()
# v1: 2.355711057750103  * x^2 +  1.8735928857661603  * x +  2.546076701607165
# v2: 6137775.166082972  * x^2 +  -3068885.398847799  * x +  3.2030852794339957
# v3: 2.433857341980365  * x^2 +  1.8882147473424797  * x +  2.511787329471587
