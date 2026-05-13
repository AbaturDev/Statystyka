import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "insurance.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())

df = df.dropna()

# =========================
# Zależność nieliniowa
# age -> charges
# =========================

X = df[["age"]]
y = df["charges"]

# =========================
# Wykres danych
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(X, y)

plt.xlabel("Wiek")
plt.ylabel("Koszt ubezpieczenia")
plt.title("Wiek vs koszt ubezpieczenia")

plt.grid(True)

plt.savefig(
    os.path.join("images", "nonlinear_relationship.png")
)

plt.show()

# =========================
# Model liniowy
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

r2_linear = r2_score(
    y_test,
    y_pred_linear
)

rmse_linear = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_linear
    )
)

print("===== MODEL LINIOWY =====")

print(f"R2: {r2_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")

# =========================
# Transformacja kwadratowa
# =========================

X_quad = X.copy()

X_quad["age_squared"] = X_quad["age"] ** 2

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
    X_quad,
    y,
    test_size=0.3,
    random_state=42
)

quad_model = LinearRegression()

quad_model.fit(X_train_q, y_train_q)

y_pred_quad = quad_model.predict(X_test_q)

r2_quad = r2_score(
    y_test_q,
    y_pred_quad
)

rmse_quad = np.sqrt(
    mean_squared_error(
        y_test_q,
        y_pred_quad
    )
)

print("\n===== MODEL Z TRANSFORMACJĄ =====")

print(f"R2: {r2_quad:.4f}")
print(f"RMSE: {rmse_quad:.4f}")

# =========================
# Porównanie modeli
# =========================

print("\n===== PORÓWNANIE =====")

print(f"R2 liniowy: {r2_linear:.4f}")
print(f"R2 transformacja: {r2_quad:.4f}")

print(f"RMSE liniowy: {rmse_linear:.4f}")
print(f"RMSE transformacja: {rmse_quad:.4f}")

if r2_quad > r2_linear:
    print("\nTransformacja poprawiła model")
else:
    print("\nTransformacja nie poprawiła modelu")

# =========================
# Wykres dopasowania
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(X_test["age"], y_test)

sorted_idx = np.argsort(X_test["age"].values.flatten())

plt.plot(
    X_test["age"].values.flatten()[sorted_idx],
    y_pred_linear[sorted_idx]
)

plt.plot(
    X_test["age"].values.flatten()[sorted_idx],
    y_pred_quad[sorted_idx]
)

plt.xlabel("Wiek")
plt.ylabel("Koszt ubezpieczenia")
plt.title("Porównanie modeli")

plt.grid(True)

plt.savefig(
    os.path.join("images", "model_comparison.png")
)

plt.show()