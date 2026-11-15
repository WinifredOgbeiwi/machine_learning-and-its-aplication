# Lab 2: Tasks 1, 3, 4, 5, & 6

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)

# --- TASK 1: Load & Preprocess Data ---
print("--- TASK 1 ---")
X, y = load_digits(return_X_y=True)
X, Y_oh = X / 16.0, np.eye(10)[y]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y_oh, test_size=0.3, stratify=y, random_state=42
)
print(
    f"Data loaded successfully. Train shape: {X_train.shape}, Test shape:"
    f" {X_test.shape}\n"
)


# --- Functions ---
def relu(z):
  return np.maximum(0, z)


def softmax(z):
  e = np.exp(z - np.max(z, axis=1, keepdims=True))
  return e / np.sum(e, axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2, W3, b3):
  return softmax(relu(relu(X @ W1 + b1) @ W2 + b2) @ W3 + b3)


def cross_entropy_loss(Y_hat, Y_true):
  return -np.sum(Y_true * np.log(np.clip(Y_hat, 1e-15, 1.0))) / Y_hat.shape[0]


# --- TASK 3 & TASK 4: Sanity Checks ---
print("--- TASK 3 & TASK 4 ---")
batch_size = 32
X_batch, Y_true_batch = X_train[:batch_size], y_train[:batch_size]

for W in [16, 64]:
  W1, b1 = np.random.normal(0, 0.01, (64, W)), np.zeros(W)
  W2, b2 = np.random.normal(0, 0.01, (W, W)), np.zeros(W)
  W3, b3 = np.random.normal(0, 0.01, (W, 10)), np.zeros(10)

  Y_hat = forward(X_batch, W1, b1, W2, b2, W3, b3)
  loss = cross_entropy_loss(Y_hat, Y_true_batch)

  assert Y_hat.shape == (32, 10) and np.allclose(
      np.sum(Y_hat, axis=1), 1.0, atol=1e-6
  )
  print(
      f"W={W}: Task 3 Shape/Row-sum OK | Task 4 Initial Loss={loss:.4f}"
      " (Expected ~2.3026)"
  )
print()


# --- TASK 5: Gradient Check (W = 16) ---
print("--- TASK 5 ---")
W = 16
W1, b1 = np.random.normal(0, 0.01, (64, W)), np.zeros(W)
W2, b2 = np.random.normal(0, 0.01, (W, W)), np.zeros(W)
W3, b3 = np.random.normal(0, 0.01, (W, 10)), np.zeros(10)


def backward(X, Y, W1, b1, W2, b2, W3, b3):
  Z1, A1 = X @ W1 + b1, relu(X @ W1 + b1)
  Z2, A2 = A1 @ W2 + b2, relu(A1 @ W2 + b2)
  Z3 = A2 @ W3 + b3
  Y_hat = softmax(Z3)

  dZ3 = (Y_hat - Y) / X.shape[0]
  dW3, db3 = A2.T @ dZ3, np.sum(dZ3, axis=0)
  dZ2 = (dZ3 @ W3.T) * (Z2 > 0)
  dW2, db2 = A1.T @ dZ2, np.sum(dZ2, axis=0)
  dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
  dW1, db1 = X.T @ dZ1, np.sum(dZ1, axis=0)
  return dW1, db1, dW2, db2, dW3, db3


grads = backward(X_batch, Y_true_batch, W1, b1, W2, b2, W3, b3)
params, names = [W1, b1, W2, b2, W3, b3], ["W1", "b1", "W2", "b2", "W3", "b3"]


def get_loss():
  return cross_entropy_loss(
      forward(X_batch, W1, b1, W2, b2, W3, b3), Y_true_batch
  )


eps, passed = 1e-5, 0
for _ in range(10):
  p_idx = np.random.randint(len(params))
  param, grad = params[p_idx], grads[p_idx]
  idx = tuple(np.random.randint(0, d) for d in param.shape)

  orig = param[idx]
  param[idx] = orig + eps
  loss_hi = get_loss()
  param[idx] = orig - eps
  loss_lo = get_loss()
  param[idx] = orig

  num_grad = (loss_hi - loss_lo) / (2 * eps)
  rel_err = np.abs(grad[idx] - num_grad) / (
      np.abs(grad[idx]) + np.abs(num_grad) + 1e-8
  )
  if rel_err < 1e-6:
    passed += 1

print(f"Gradient Check Result: {passed}/10 passed (< 1e-6 requirement met).\n")
assert passed >= 9


# --- TASK 6: Full Training Loop & Hyperparameter Sweeps ---
# --- TASK 6: Full Training Loop, Hyperparameter Sweeps & Loss Curve Plotting ---
print("--- TASK 6 ---")
import os

os.makedirs("results", exist_ok=True)

epochs, learning_rates, hidden_widths = 20, [0.001, 0.01, 0.1], [16, 64]
plt.figure(figsize=(12, 8))

for idx, W in enumerate(hidden_widths):
  for lr in learning_rates:
    np.random.seed(42)
    W1, b1 = np.random.normal(0, 0.01, (64, W)), np.zeros(W)
    W2, b2 = np.random.normal(0, 0.01, (W, W)), np.zeros(W)
    W3, b3 = np.random.normal(0, 0.01, (W, 10)), np.zeros(10)

    epoch_losses = []

    for epoch in range(epochs):
      indices = np.random.permutation(X_train.shape[0])
      X_shuff, y_shuff = X_train[indices], y_train[indices]

      batch_losses = []
      for i in range(0, X_train.shape[0], batch_size):
        X_b, y_b = X_shuff[i : i + batch_size], y_shuff[i : i + batch_size]

        # Forward & Backward
        dW1, db1, dW2, db2, dW3, db3 = backward(X_b, y_b, W1, b1, W2, b2, W3, b3)

        # Compute batch loss for tracking
        Y_hat = forward(X_b, W1, b1, W2, b2, W3, b3)
        batch_losses.append(cross_entropy_loss(Y_hat, y_b))

        # Update
        W1, b1 = W1 - lr * dW1, b1 - lr * db1
        W2, b2 = W2 - lr * dW2, b2 - lr * db2
        W3, b3 = W3 - lr * dW3, b3 - lr * db3

      epoch_losses.append(np.mean(batch_losses))

    # Plotting curve
    plt.plot(
        range(1, epochs + 1),
        epoch_losses,
        label=f"Width={W}, LR={lr}",
        marker="o",
        markersize=3,
    )

    # Test Accuracy evaluation
    test_preds = np.argmax(forward(X_test, W1, b1, W2, b2, W3, b3), axis=1)
    test_acc = np.mean(test_preds == np.argmax(y_test, axis=1)) * 100.0
    print(
        f"Width W={W}, LR={lr} | Final Train Loss: {epoch_losses[-1]:.4f} | Test"
        f" Accuracy: {test_acc:.2f}%"
    )

plt.title("Training Loss Curves Across Hyperparameters")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("lab02/results/loss_curves.png", dpi=300, bbox_inches="tight")
plt.close()


# --- TASK 7: Time Forward and Backward Passes Separately (Median of 100 runs) ---
import time

print("--- TASK 7 ---")


_ = forward(X_batch, W1, b1, W2, b2, W3, b3)
_ = backward(X_batch, Y_true_batch, W1, b1, W2, b2, W3, b3)


forward_times = []
for _ in range(100):
  start = time.perf_counter()
  _ = forward(X_batch, W1, b1, W2, b2, W3, b3)
  forward_times.append(time.perf_counter() - start)


backward_times = []
for _ in range(100):
  start = time.perf_counter()
  _ = backward(X_batch, Y_true_batch, W1, b1, W2, b2, W3, b3)
  backward_times.append(time.perf_counter() - start)

median_fwd = np.median(forward_times) * 1000 
median_bwd = np.median(backward_times) * 1000 

print(f"Median Forward Pass Time: {median_fwd:.4f} ms")
print(f"Median Backward Pass Time: {median_bwd:.4f} ms")
print(f"Ratio (Backward/Forward): {median_bwd / median_fwd:.2f}x\n")