# Lab 01: Environment and First System Measurements

## 1. Goal
* The goal of this lab is to build a reproducible Python environment, train two baseline models (Logistic Regression and Random Forest) on the Breast Cancer Wisconsin dataset, measure their real-world computing costs (speed, file size, and RAM), and figure out which hardware platforms (Cloud, Edge, Mobile, or TinyML) can run them.

## 2. Method
* **Setup & Versions:** I created an isolated virtual environment (`.venv`) and installed pinned packages from `requirements.txt`. My environment runs on `[Your OS, e.g., Windows 11]` with Python `[Paste Python version from results/versions.txt]` and scikit-learn `[Paste scikit-learn version from results/versions.txt]`.
* **Data & Models:** I loaded the Breast Cancer Wisconsin dataset (`569` samples, `30` features) and split it 70/30 using a fixed random seed (`42`). I trained a linear model (`LogisticRegression` with `StandardScaler`) and a tree ensemble (`RandomForestClassifier` with 100 trees).
* **Measurements:** To get reliable numbers, I ran 1 warm-up round and then took the median of 5 runs for training time, and the median of 100 runs for single-sample prediction speed. I checked model file size using `joblib.dump` and measured peak RAM usage using `psutil`.

## 3. Results

### Model Accuracy & System Costs
| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **Test Accuracy** | `0.9883` *(check your `baseline_accuracy.csv`)* | `0.9357` *(check your `baseline_accuracy.csv`)* |
| **Training Time (median of 5)** | `6.8730 ms` | `113.8087 ms` |
| **Single-Sample Speed (median of 100)** | `0.1385 ms` | `2.0651 ms` |
| **Model File Size** | `2,286 bytes` (`2.23 KB`) | `290,889 bytes` (`284.07 KB`) |
| **Peak RAM Used (`psutil` RSS)** | `148.53 MB` | `149.00 MB` |

### Deployment Platform Fit
| Platform | Max RAM[cite: 2] | Max Speed[cite: 2] | Max Size[cite: 2] | Logistic Regression | Random Forest |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cloud** | $\ge 1\text{ GB}$ | $\le 100\text{ ms}$ | $\le 500\text{ MB}$ | **Fits** | **Fits** |
| **Edge** | $256\text{–}1024\text{ MB}$ | $\le 50\text{ ms}$ | $\le 50\text{ MB}$ | **Fits** | **Fits** |
| **Mobile** | $64\text{–}256\text{ MB}$ | $\le 20\text{ ms}$ | $\le 10\text{ MB}$ | **Fits** | **Fits** |
| **TinyML** | $\le 256\text{ KB}$ | $\le 10\text{ ms}$ | $\le 100\text{ KB}$ | **Fails** | **Fails (Too big)** |

**Why they fit or fail:**
* **Cloud, Edge, and Mobile:** Both models easily pass all memory, speed, and file-size limits on these three platforms. However, Logistic Regression is much better for Mobile because it is ~15× faster (`0.14 ms` vs. `2.07 ms`) and saves battery.
* **TinyML:** Random Forest fails immediately because its file size (`284.07 KB`) is almost 3× larger than the `100 KB` limit. Logistic Regression also fails as a Python program because Python itself uses `148.53 MB` of RAM (way over `256 KB`), **but** since the model file is only `2.23 KB` and takes `0.14 ms`, it can fit on TinyML if exported to C without Python.

## 4. Conclusions
1. **Simpler models can be both more accurate and much cheaper:** On this dataset, Logistic Regression beat Random Forest in accuracy (`0.9883` vs. `0.9357`) while running ~15× faster and taking ~127× less storage space.
2. **Python takes up most of the RAM:** Both models used nearly the same amount of RAM (~148–149 MB) because the Python interpreter and libraries take up ~148 MB before the model even does any work.
3. **Where a model runs depends on the software runtime, not just the model size:** Random Forest is too large on disk (`284 KB`) for a TinyML chip (`100 KB` max), while Logistic Regression (`2.23 KB`) is small enough to fit—as long as we remove Python and run it in C.