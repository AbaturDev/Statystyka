import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "BostonHousing.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())

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
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R2: {r2:.4f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)

print("Intercept:", model.intercept_)

X_sm = sm.add_constant(X)

ols_model = sm.OLS(y, X_sm).fit()

print(ols_model.summary())

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Rzeczywiste ceny")
plt.ylabel("Przewidywane ceny")
plt.title("Rzeczywiste vs przewidywane")
plt.grid(True)
plt.savefig(os.path.join("images", "predicted_vs_actual.png"))
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Przewidywane wartości")
plt.ylabel("Reszty")
plt.title("Wykres reszt")
plt.grid(True)
plt.savefig(os.path.join("images", "residuals_plot.png"))
plt.show()

correlation_matrix = X.corr()

print(correlation_matrix)