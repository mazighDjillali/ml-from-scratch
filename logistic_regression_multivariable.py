import numpy as np

X = np.array([
    [0.5, 4],
    [1.0, 5],
    [1.5, 4],
    [2.0, 5],
    [2.5, 8],   # peu revise MAIS bien dormi  -> recu
    [3.0, 2],   # plus revise MAIS mal dormi  -> recale
    [3.5, 7],
    [4.0, 6],
    [4.5, 7],
    [5.0, 6],
])
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.zeros(2)  # Initialisation des poids pour deux caractéristiques
b = 0
lr = 0.05
nb_iteration = 5000

for n in range(nb_iteration):
    z = X @ w + b
    y_pred = sigmoid(z)
    erreur = y_pred - y
    grad_w = X.T @ erreur / len(y)  # Gradient pour les poids
    grad_b = np.mean(erreur)

    w -= lr * grad_w
    b -= lr * grad_b

print("Poids w1:", w[0])
print("Poids w2:", w[1])
print("Biais:", b)
z = X @ w + b
y_pred = sigmoid(z)

resultat = np.where(y_pred > 0.5, 1, 0)
print("Prédictions:", resultat)
print("Exactitude:", np.mean(resultat == y))
