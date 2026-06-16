# Day 11 – AI Internship Report

## Implementing a 2-Layer Neural Network from Scratch (NumPy)

---

## 📌 Objective

The goal of Day 11 was to build a **fully functional 2-layer neural network from scratch using only NumPy**, in order to deeply understand how deep learning works internally, especially:

* Forward propagation
* Activation functions (ReLU, Sigmoid)
* Loss computation (Binary Cross-Entropy)
* Backpropagation using chain rule
* Gradient descent optimization
* Decision boundary visualization

No deep learning frameworks (TensorFlow/PyTorch/Keras) were used.

---

## 📊 Dataset Used

We used the **`make_moons` dataset** from Scikit-learn:

* 500 samples
* 2 input features
* Binary classification (0/1)
* Non-linear structure (moon-shaped patterns)

This dataset is ideal for testing neural networks because it cannot be separated using a simple linear model.

---

## 🧠 Model Architecture

A simple feedforward neural network was implemented:

```
Input Layer  →  Hidden Layer  →  Output Layer
(2 features)     (8 neurons)      (1 neuron)
```

### Parameters:

* Weights:

  * W1: (2 × 8)
  * W2: (8 × 1)

* Biases:

  * b1: (1 × 8)
  * b2: (1 × 1)

---

## ⚙️ Key Concepts Implemented

### 1. Forward Propagation

Computed predictions step-by-step:

```
Z1 = X·W1 + b1
A1 = ReLU(Z1)
Z2 = A1·W2 + b2
A2 = Sigmoid(Z2)
```

---

### 2. Activation Functions

* **ReLU (Hidden Layer)**

  ```
  f(x) = max(0, x)
  ```

* **Sigmoid (Output Layer)**
  Converts outputs into probabilities (0 to 1)

---

### 3. Loss Function

Binary Cross-Entropy Loss was used:

```
Loss = -mean[y log(p) + (1 - y) log(1 - p)]
```

This measures how far predictions are from actual labels.

---

### 4. Backpropagation

Implemented manually using chain rule:

* Computed gradients for output layer
* Propagated error backward to hidden layer
* Calculated gradients for all weights and biases

---

### 5. Gradient Descent

Weights updated using:

```
W = W - learning_rate × gradient
```

This allowed the model to gradually minimize loss.

---

## 📉 Training Process

* Epochs: 1000
* Learning Rate: 0.1

During training:

* Loss steadily decreased from ~0.69
* Model gradually learned non-linear decision boundaries

---

## 📈 Results

### ✔ Loss Curve

* Shows steady reduction in error over epochs
* Confirms successful learning process

### ✔ Decision Boundary

* Initially random
* Eventually adapts to separate the two moon-shaped classes

---

## 🔍 Key Learnings

* Neural networks are just layered mathematical functions
* Backpropagation is based on the chain rule
* Learning happens through small weight adjustments
* Non-linear activation functions are essential
* Even simple networks can solve complex patterns

---

## 🚀 Conclusion

This project successfully demonstrated how a neural network learns from scratch without any deep learning libraries. It provided a strong foundation in:

* Neural network internals
* Forward & backward propagation
* Gradient-based optimization

This marks a major step toward understanding modern deep learning systems.

---

## 📌 Next Steps

* Extend to deeper neural networks
* Implement mini-batch gradient descent
* Train on MNIST dataset
* Convert implementation to PyTorch for comparison

---
