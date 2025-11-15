import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


X = np.array([
    [0.95, 0.8, 0.9, 1],
    [0.85, 0.6, 0.7, 1],
    [0.70, 0.4, 0.6, 0],
    [0.60, 0.3, 0.4, 0],
    [0.40, 0.2, 0.3, 0]
])


y = np.array([1, 1, 0, 0, 0])


lr = LogisticRegression()
lr.fit(X, y)


joblib.dump(lr, "ranker_model.pkl")

print("✅ ranker_model.pkl created successfully!")
