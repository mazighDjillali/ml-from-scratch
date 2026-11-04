import numpy as np

# Donnees : X = heures revisees avant un exam, y = reussite(1) / echec(0)
X = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y = np.array([0,   0,   0,   0,   1,   0,   1,   1,   1,   1  ])

# 🟢 je te fournis la sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

nb_iteration = 5000
w=0
b=0
lr = 0.05
for i in range(nb_iteration):
    z = w * X + b
    y_pred = sigmoid(z)
    erreur = y_pred - y
    grad_w = np.mean(erreur * X)
    grad_b = np.mean(erreur)

    w = w - grad_w * lr
    b= b - grad_b*lr

#    puis affiche w, b finaux + l'exactitude (part de predictions correctes).
print(w)
z = w * X + b
y_pred = sigmoid(z)
print(y_pred)
print('w', w)
resultat = np.where(y_pred > 0.5, 1, 0)
print(resultat)

print ("Exactitude : ", np.mean(resultat == y))
