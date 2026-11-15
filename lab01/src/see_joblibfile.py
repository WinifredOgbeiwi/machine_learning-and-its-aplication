import joblib
from sklearn.datasets import load_breast_cancer

model = joblib.load("./lab01/results/model.joblib")
X, y = load_breast_cancer(return_X_y=True)

print("Model type:", type(model).__name__)
print("Number of trees stored:", len(model.estimators_))
print("Number of input features learned:", model.n_features_in_)
print("Sample prediction on row 0:", model.predict(X[0:1])[0])