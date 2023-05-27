from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# incarc cifrele
cifre = np.loadtxt("sport.csv", delimiter=",")
realOutputs = np.array(cifre[:, :-3])
computedOutputs = np.array(cifre[:, -3:])

# plotez cifrele
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))

for i, ax in enumerate(axes):
    real, = ax.plot(realOutputs[:, i], 'ro', label='real')
    computed, = ax.plot(computedOutputs[:, i], 'bo', label='computed')
    ax.set_xlim(-0.5, len(realOutputs) - 0.5)
    ax.set_ylim(np.min(realOutputs[:, i]) - 1, np.max(realOutputs[:, i]) + 1)
    ax.legend([real, computed], ["Real", "Computed"])
    ax.set_xlabel('Sample')
    ax.set_ylabel(['Weight', 'Waist', 'Pulse'][i])

plt.show()

# calculez eroarea de pr

# MAE
errorsL1 = [sum(abs(r - c) for r, c in zip(realOutputs[:, i], computedOutputs[:, i]))
            / len(realOutputs) for i in range(3)]
errorL1 = sum(errorsL1) / 3
print('Error (L1): ', errorL1)

# RMSE
errorsL2 = [sqrt(sum((r - c) ** 2 for r, c in zip(realOutputs[:, i], computedOutputs[:, i]))
                 / len(realOutputs)) for i in range(3)]
errorL2 = sum(errorsL2) / 3
print('Error (L2): ', errorL2)



