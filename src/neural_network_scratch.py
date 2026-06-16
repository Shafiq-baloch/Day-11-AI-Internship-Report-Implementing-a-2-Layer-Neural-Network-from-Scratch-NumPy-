# ==========================================================
# DAY 11 - 2-LAYER NEURAL NETWORK FROM SCRATCH
# Libraries: NumPy + Matplotlib only
# Dataset: make_moons
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ==========================================================
# 1. DATASET
# ==========================================================

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# reshape y to (m,1)
y = y.reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# visualize dataset
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm")
plt.title("Make Moons Dataset")
plt.show()

# ==========================================================
# 2. NEURAL NETWORK CLASS
# ==========================================================

class NeuralNetwork:

    def __init__(self, lr=0.1):

        np.random.seed(42)

        # learning rate
        self.lr = lr

        # weights
        self.W1 = np.random.randn(2, 8) * 0.01
        self.b1 = np.zeros((1, 8))

        self.W2 = np.random.randn(8, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    # -------------------------
    # ACTIVATIONS
    # -------------------------

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    # -------------------------
    # FORWARD PASS
    # -------------------------

    def forward(self, X):

        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    # -------------------------
    # LOSS FUNCTION
    # -------------------------

    def compute_loss(self, y, y_hat):

        m = y.shape[0]
        eps = 1e-15

        y_hat = np.clip(y_hat, eps, 1 - eps)

        loss = -np.mean(
            y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        )

        return loss

    # -------------------------
    # BACKPROPAGATION
    # -------------------------

    def backward(self, X, y, y_hat):

        m = y.shape[0]

        # output layer error
        dZ2 = y_hat - y
        self.W2_grad = np.dot(self.A1.T, dZ2) / m
        self.b2_grad = np.sum(dZ2, axis=0, keepdims=True) / m

        # hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)

        self.W1_grad = np.dot(X.T, dZ1) / m
        self.b1_grad = np.sum(dZ1, axis=0, keepdims=True) / m

    # -------------------------
    # UPDATE PARAMETERS
    # -------------------------

    def update(self):

        self.W1 -= self.lr * self.W1_grad
        self.b1 -= self.lr * self.b1_grad

        self.W2 -= self.lr * self.W2_grad
        self.b2 -= self.lr * self.b2_grad

    # -------------------------
    # TRAIN FUNCTION
    # -------------------------

    def train(self, X, y, epochs=1000):

        losses = []

        for i in range(epochs):

            # forward
            y_hat = self.forward(X)

            # loss
            loss = self.compute_loss(y, y_hat)
            losses.append(loss)

            # backward
            self.backward(X, y, y_hat)

            # update weights
            self.update()

            # print progress
            if i % 100 == 0:
                print(f"Epoch {i} | Loss: {loss}")

        return losses

    # -------------------------
    # PREDICT
    # -------------------------

    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat > 0.5).astype(int)

# ==========================================================
# 3. TRAIN MODEL
# ==========================================================

nn = NeuralNetwork(lr=0.1)

losses = nn.train(X, y, epochs=1000)

# ==========================================================
# 4. LOSS PLOT
# ==========================================================

plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# ==========================================================
# 5. DECISION BOUNDARY
# ==========================================================

def plot_decision_boundary(model, X, y):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.forward(grid)
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolors="k")
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(nn, X, y)