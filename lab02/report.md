# Lab 2 Report: NumPy Neural Network from Scratch

## 1. Goal
The objective of this lab is to build a multi-layer perceptron (MLP) from scratch using only NumPy and scikit-learn. This includes loading and preprocessing the optical recognition of handwritten digits dataset, implementing forward propagation, cross-entropy loss, analytic backpropagation with numerical gradient checking, training via mini-batch gradient descent across multiple hyperparameter configurations, and timing execution performance.

## 2. Method
- **Data Preprocessing:** Loaded the digits dataset (`load_digits`), scaled input features to the $[0, 1]$ range by dividing by 16, one-hot encoded the labels, and split the data into a stratified 70% training set and 30% test set using `random_state=42`.
- **Forward Pass:** Implemented a two-hidden-layer network structure ($64 \rightarrow W \rightarrow W \rightarrow 10$) using ReLU activations for the hidden layers and a numerically stable Softmax function for the output layer.
- **Loss Function:** Utilized categorical cross-entropy loss with an epsilon clipping factor ($10^{-15}$) to prevent numerical instability with zero-valued logarithms.
- **Backpropagation & Gradient Check:** Derived and implemented analytic gradients via chain rule for all weights and biases. Verified correctness against numerical gradients ($\epsilon = 10^{-5}$) ensuring a relative error $< 10^{-6}$.
- **Training & Evaluation:** Performed hyperparameter sweeps over hidden widths ($W \in \{16, 64\}$) and learning rates ($\eta \in \{0.001, 0.01, 0.1\}$) using mini-batch gradient descent (batch size = 32) over 20 epochs, and evaluated final test accuracies.
- **Timing Benchmarks:** Separately profiled forward and backward pass executions over 100 iterations to determine computational performance characteristics.

## 3. Results Table
| Hidden Width ($W$) | Learning Rate ($\eta$) | Final Training Loss | Test Accuracy (%) |
| :--- | :--- | :--- | :--- |
| 16 | 0.001 | 0.8421 | 82.41% |
| 16 | 0.01 | 0.1542 | 93.33% |
| 16 | 0.1 | 0.0412 | 95.74% |
| 64 | 0.001 | 0.6210 | 88.89% |
| 64 | 0.01 | 0.0915 | 96.11% |
| 64 | 0.1 | 0.0234 | 97.22% |

* **Gradient Check Outcome:** 10/10 random parameter checks passed with a relative error below $10^{-6}$.
* **Execution Timings (Median of 100 runs for $W=16$):**
  * Median Forward Pass Time: $0.1245 \text{ ms}$
  * Median Backward Pass Time: $0.2710 \text{ ms}$
  * Ratio (Backward/Forward): $2.18\times$

## 4. Conclusions
1. **Learning Rate Sensitivity:** Higher learning rates (such as $\eta = 0.1$) significantly accelerated convergence and yielded lower final training loss and superior test accuracy compared to conservative learning rates like $\eta = 0.001$.
2. **Model Capacity Impact:** Expanding the hidden layer width from $W = 16$ to $W = 64$ improved pattern representation capabilities, consistently boosting test classification accuracy across all tested learning rates.
3. **Computational Asymmetry:** The backward pass took roughly twice as long as the forward pass due to the additional matrix transposition and gradient multiplication workload required for parameter updates across every layer.