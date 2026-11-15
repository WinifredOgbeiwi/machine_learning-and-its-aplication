import os
import random
import time
import joblib
import psutil
import random
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)

# TASK 2: Train baseline models and save test accuracy to CSV
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

records = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    records.append({"model": name, "test_accuracy": f"{acc:.4f}"})


df = pd.DataFrame(records)
df.to_csv("lab01/results/baseline_accuracy.csv", index=False)


# TASK 3: Measure training time, inference time, model size, and peak memory usage
process = psutil.Process(os.getpid())
single_sample = X_test[0:1]
model_path = "lab01/results/model.joblib"

for name, base_model in models.items():
  
    clone(base_model).fit(X_train, y_train)
    train_times_ms = []
    peak_rss_bytes = process.memory_info().rss

    for _ in range(5):
        m = clone(base_model)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        train_times_ms.append((time.perf_counter() - t0) * 1000.0)
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

    median_train_ms = float(np.median(train_times_ms))

   
    fitted_model = clone(base_model)
    fitted_model.fit(X_train, y_train)

    
    _ = fitted_model.predict(single_sample)
    infer_times_ms = []

    for _ in range(100):
        t0 = time.perf_counter()
        _ = fitted_model.predict(single_sample)
        infer_times_ms.append((time.perf_counter() - t0) * 1000.0)
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

    median_infer_ms = float(np.median(infer_times_ms))

  
    joblib.dump(fitted_model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    
    peak_rss_mb = peak_rss_bytes / (1024.0 ** 2)

    print(f"=== {name} ===")
    print(f"Training time (median of 5):       {median_train_ms:.4f} ms")
    print(f"Single-sample inference (med 100): {median_infer_ms:.4f} ms")
    print(f"Model size:                        {size_bytes} bytes ({size_kb:.2f} KB)")
    print(f"Peak RSS memory:                   {peak_rss_mb:.2f} MB\n")