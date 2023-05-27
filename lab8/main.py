import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from file import reader
from normalization import normalize
from MyLogisticRegression import MyLogisticRegression
from file.profa import main2
import matplotlib.pyplot as plt
import numpy as np


def main():
    intrari, carac1, carac2, carac3, carac4, iesiri, iesireNume = reader.read()

    labels = set(iesiri)
    noData = len(intrari)
    for crtLabel in labels:
        x1 = [carac1[i] for i in range(noData) if iesiri[i] == crtLabel]
        x2 = [carac2[i] for i in range(noData) if iesiri[i] == crtLabel]
        x3 = [carac3[i] for i in range(noData) if iesiri[i] == crtLabel]
        x4 = [carac4[i] for i in range(noData) if iesiri[i] == crtLabel]

        plt.scatter(x1, x2, x3, x4, label=iesireNume[crtLabel])
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.legend()
    plt.show()

    def plotClassificationData(carac1, carac2, carac3, carac4, iesiri, title=None):
        labels = set(iesiri)
        noData = len(carac1)

        for crtLabel in labels:
            x = [carac1[i] for i in range(noData) if iesiri[i] == crtLabel]
            y = [carac2[i] for i in range(noData) if iesiri[i] == crtLabel]
            s = [carac3[i] for i in range(noData) if iesiri[i] == crtLabel]
            c = [carac4[i] for i in range(noData) if iesiri[i] == crtLabel]

            # Check for negative or invalid size values
            valid_sizes = [size for size in s if size > 0]

            if len(valid_sizes) > 0:
                # Increase the size of the circles by multiplying with a scaling factor (e.g., 10)
                scaled_sizes = [size * 10 for size in s]
                plt.scatter(x, y, s=scaled_sizes, c=c, label=iesireNume[crtLabel])

        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')
        plt.legend()
        plt.title(title)
        plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(4 * 3, 4))
    ax[0].hist(carac1, 10)
    ax[0].title.set_text('Histogram of sepal length (cm)')
    ax[1].hist(carac2, 10)
    ax[1].title.set_text('Histogram of sepal width (cm)')
    ax[2].hist(iesiri, 10)
    ax[2].title.set_text('Histogram of iris class')
    plt.show()

    np.random.seed(5)
    indexes = [i for i in range(len(intrari))]
    train_sample = np.random.choice(indexes, int(0.8 * len(intrari)), replace=False)
    test_sample = [i for i in indexes if not i in train_sample]

    train_intrari = [intrari[i] for i in train_sample]
    train_iesiri = [iesiri[i] for i in train_sample]
    test_intrari = [intrari[i] for i in test_sample]
    test_iesiri = [iesiri[i] for i in test_sample]

    train_intrari, test_intrari = normalize(train_intrari, test_intrari)

    # plot the normalised data
    carac1train = [ex[0] for ex in train_intrari]
    carac2train = [ex[1] for ex in train_intrari]
    carac3train = [ex[2] for ex in train_intrari]
    carac4train = [ex[3] for ex in train_intrari]
    carac1test = [ex[0] for ex in test_intrari]
    carac2test = [ex[1] for ex in test_intrari]
    carac3test = [ex[2] for ex in test_intrari]
    carac4test = [ex[3] for ex in test_intrari]

    plotClassificationData(carac1train, carac2train, carac3train, carac4train, train_iesiri, 'normalised train data')

    labels = [label for label in set(iesiri)]
    classifier = linear_model.LogisticRegression()
    classifier.fit(train_intrari, train_iesiri)
    w0 = classifier.intercept_
    w1, w2, w3, w4 = [classifier.coef_[_][0] for _ in range(len(labels))], [classifier.coef_[_][1] for _ in range(len(labels))], [classifier.coef_[_][2] for _ in range(len(labels))], [classifier.coef_[_][3] for _ in range(len(labels))]
    print('Classification models: (using tool)\n')
    for _ in range(len(labels)):
        print('y' + str(_ + 1) + '(feat1, feat2, feat3, feat4) = ', w0[_], ' + ', w1[_], ' * feat1 + ', w2[_], ' * feat2 + ', w3[_], ' * feat3 + ', w4[_], ' * feat4\n')

    classifier = MyLogisticRegression()
    classifier.fit(train_intrari, train_iesiri)
    w0 = classifier.intercept_
    w1, w2, w3, w4 = [classifier.coef_[i][0] for i in range(len(labels))], [classifier.coef_[i][1] for i in range(len(labels))], [classifier.coef_[i][2] for i in range(len(labels))], [classifier.coef_[i][3] for i in range(len(labels))]
    print('Classification models: (using developed code)\n')
    for _ in range(len(labels)):
        print('y' + str(_ + 1) + '(feat1, feat2, feat3, feat4) = ', w0[_], ' + ', w1[_], ' * feat1 + ', w2[_],
              ' * feat2 + ', w3[_], ' * feat3 + ', w4[_], ' * feat4\n')

    computed_test_iesiri = classifier.predict(test_intrari)

    def plotPredictions(carac1, carac2, carac3, carac4, realiesiri, computediesiri, title, labelNames):
        labels = list(set(realiesiri))
        noData = len(carac1)

        for crtLabel in labels:
            x = [carac1[i] for i in range(noData) if realiesiri[i] == crtLabel and computediesiri[i] == crtLabel]
            y = [carac2[i] for i in range(noData) if realiesiri[i] == crtLabel and computediesiri[i] == crtLabel]
            plt.scatter(x, y, label=labelNames[crtLabel] + ' (correct)', marker='o')

        for crtLabel in labels:
            x = [carac1[i] for i in range(noData) if realiesiri[i] == crtLabel and computediesiri[i] != crtLabel]
            y = [carac2[i] for i in range(noData) if realiesiri[i] == crtLabel and computediesiri[i] != crtLabel]
            plt.scatter(x, y, label=labelNames[crtLabel] + ' (incorrect)', marker='x')

        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')
        plt.legend()
        plt.title(title)
        plt.show()

    plotPredictions(carac1test, carac2test, carac3test, carac4test, test_iesiri, computed_test_iesiri, "real test data", iesireNume)
    error = 0.0
    for t1, t2 in zip(computed_test_iesiri, test_iesiri):
        if t1 != t2:
            error += 1
    error = error / len(test_iesiri)
    print("Classification error (using developed code): ", error)

    error = 1 - accuracy_score(test_iesiri, computed_test_iesiri)
    print("Classification error (using tool): ", error)


main()
#main2()