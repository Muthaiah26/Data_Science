import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---- TRAINING SAMPLE ----
# Each row = [vector_score, skill_ratio, recency_score, loc_score]
X = np.array([
    [0.95, 0.8, 0.9, 1],
    [0.85, 0.6, 0.7, 1],
    [0.70, 0.4, 0.6, 0],
    [0.60, 0.3, 0.4, 0],
    [0.40, 0.2, 0.3, 0]
])

# Label: 1 = good match, 0 = bad match
y = np.array([1, 1, 0, 0, 0])

# ---- TRAIN MODEL ----
lr = LogisticRegression()
lr.fit(X, y)

# ---- SAVE MODEL ----
joblib.dump(lr, "ranker_model.pkl")

print("✅ ranker_model.pkl created successfully!")
