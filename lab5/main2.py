import csv
import numpy as np


with open('flowers.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    rezultate = [sir for sir in reader]

denumiri = ['Daisy', 'Tulip', 'Rose']

# Create confusion matrix
confusion_matrix = np.zeros((3, 3), dtype=int)

for sir in rezultate:
    true_index = denumiri.index(sir[0])
    predicted_index = denumiri.index(sir[1])
    confusion_matrix[true_index, predicted_index] += 1

total = np.sum(confusion_matrix)
accuracy = np.sum(np.diag(confusion_matrix)) / total
precisions = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recalls = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)


# Categorical cross-entropy loss
random_probabilities = {
    'Daisy': [0.5, 0.3, 0.2],
    'Tulip': [0.3, 0.5, 0.2],
    'Rose': [0.2, 0.3, 0.5],
}

losses = []
# Average Categorical Cross-Entropy
for sir in rezultate:
    true_index = denumiri.index(sir[0])
    predicted_probs = random_probabilities[sir[1]]
    loss = -np.log(predicted_probs[true_index])
    losses.append(loss)

avg_loss = np.mean(losses)

print("Confusion Matrix:\n", confusion_matrix)
print("\nAccuracy: ", accuracy)
print("Precisions: ", precisions)
print("Recalls: ", recalls)
print("\nLoss: ", avg_loss)
