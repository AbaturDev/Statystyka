import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

path = os.path.join("data", "BostonHousing.csv")
df = pd.read_csv(path)

df = df.dropna()

TARGET = "medv"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =========================
# TRAIN
# =========================

r2_train = r2_score(y_train, y_train_pred)

mae_train = mean_absolute_error(
    y_train,
    y_train_pred
)

rmse_train = np.sqrt(
    mean_squared_error(
        y_train,
        y_train_pred
    )
)

# =========================
# TEST
# =========================

r2_test = r2_score(y_test, y_test_pred)

mae_test = mean_absolute_error(
    y_test,
    y_test_pred
)

rmse_test = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

print("===== ZBIÓR TRENINGOWY =====")

print(f"R2: {r2_train:.4f}")
print(f"MAE: {mae_train:.4f}")
print(f"RMSE: {rmse_train:.4f}")

print("\n===== ZBIÓR TESTOWY =====")

print(f"R2: {r2_test:.4f}")
print(f"MAE: {mae_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")

# =========================
# INTERPRETACJA
# =========================

print("\n===== INTERPRETACJA =====")

r2_diff = abs(r2_train - r2_test)

if r2_train > 0.9 and r2_diff > 0.1:
    print("Możliwe przeuczenie (overfitting)")
elif r2_train < 0.5 and r2_test < 0.5:
    print("Możliwe niedouczenie (underfitting)")
else:
    print("Model generalizuje poprawnie")