import os
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


path = os.path.join("data", "housing.csv")

df = pd.read_csv(path)

df = df.dropna()

TARGET = "Price"

X = df.drop(columns=[TARGET, "Address"])
y = df[TARGET]

X_vif = add_constant(X)

vif_data = pd.DataFrame()

vif_data["Feature"] = X_vif.columns

vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

vif_data = vif_data[vif_data["Feature"] != "const"]

print("===== VIF =====")
print(vif_data)

feature_to_remove = vif_data.sort_values(
    by="VIF",
    ascending=False
).iloc[0]["Feature"]

print(f"\nUsuwana zmienna: {feature_to_remove}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_before = LinearRegression()

model_before.fit(X_train, y_train)

y_pred_before = model_before.predict(X_test)

r2_before = r2_score(y_test, y_pred_before)

X_reduced = X.drop(columns=[feature_to_remove])

X_train_red, X_test_red, y_train, y_test = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    random_state=42
)

model_after = LinearRegression()

model_after.fit(X_train_red, y_train)

y_pred_after = model_after.predict(X_test_red)

r2_after = r2_score(y_test, y_pred_after)

print("\n===== PORÓWNANIE =====")

print(f"R2 przed usunięciem: {r2_before:.4f}")
print(f"R2 po usunięciu: {r2_after:.4f}")

X_reduced_vif = add_constant(X_reduced)

new_vif = pd.DataFrame()

new_vif["Feature"] = X_reduced_vif.columns

new_vif["VIF"] = [
    variance_inflation_factor(X_reduced_vif.values, i)
    for i in range(X_reduced_vif.shape[1])
]

new_vif = new_vif[new_vif["Feature"] != "const"]

print("\n===== NOWE VIF =====")

print(new_vif)