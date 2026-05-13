import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "BostonHousing.csv")
df = pd.read_csv(path)

df = df.dropna()

TARGET = "medv"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

print(f"R2: {r2:.4f}")

residuals = y_test - y_pred

# =========================
# WYKRES RESZT
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Przewidywane wartości")
plt.ylabel("Reszty")
plt.title("Reszty vs przewidywania")

plt.grid(True)

plt.savefig(
    os.path.join("images", "residuals_analysis.png")
)

plt.show()

# =========================
# TEST SHAPIRO-WILKA
# =========================

shapiro_stat, shapiro_p = shapiro(residuals)

print("\n===== TEST SHAPIRO-WILKA =====")
print(f"Statystyka: {shapiro_stat:.4f}")
print(f"p-value: {shapiro_p:.4f}")

if shapiro_p > 0.05:
    print("Reszty mają rozkład normalny")
else:
    print("Reszty nie mają rozkładu normalnego")

# =========================
# TEST BREUSCHA-PAGANA
# =========================

X_test_sm = sm.add_constant(X_test)

bp_test = het_breuschpagan(
    residuals,
    X_test_sm
)

labels = [
    "LM Statistic",
    "LM-Test p-value",
    "F-Statistic",
    "F-Test p-value"
]

print("\n===== TEST BREUSCHA-PAGANA =====")

for label, value in zip(labels, bp_test):
    print(f"{label}: {value:.4f}")

if bp_test[1] > 0.05:
    print("Brak heteroskedastyczności")
else:
    print("Występuje heteroskedastyczność")

# =========================
# NIEZALEŻNOŚĆ RESZT
# =========================

dw_stat = durbin_watson(residuals)

print("\n===== TEST DURBINA-WATSONA =====")
print(f"Statystyka: {dw_stat:.4f}")

if 1.5 <= dw_stat <= 2.5:
    print("Reszty są niezależne")
else:
    print("Możliwa autokorelacja reszt")