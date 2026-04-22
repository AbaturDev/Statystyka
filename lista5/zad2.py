import os
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

path = os.path.join("data", "housing.csv")
df = pd.read_csv(path)

df = df.dropna()

TARGET = "Price"

X = df.drop(columns=[TARGET, "Address"])
y = df[TARGET]

X_full = sm.add_constant(X)

full_model = sm.OLS(y, X_full).fit()

print("===== MODEL PEŁNY =====")
print(full_model.summary())

X_backward = X.copy()

while True:

    X_with_const = sm.add_constant(X_backward)

    model = sm.OLS(y, X_with_const).fit()

    p_values = model.pvalues.drop("const")

    max_p = p_values.max()

    if max_p > 0.05:

        feature_to_remove = p_values.idxmax()

        print(f"Usuwanie zmiennej: {feature_to_remove} (p = {max_p})")

        X_backward = X_backward.drop(columns=[feature_to_remove])

    else:
        break


X_reduced = sm.add_constant(X_backward)

reduced_model = sm.OLS(y, X_reduced).fit()

print("\n===== MODEL ZREDUKOWANY =====")
print(reduced_model.summary())

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train)

y_pred_full = model_full.predict(X_test_full)

r2_full = r2_score(y_test, y_pred_full)


X_train_red, X_test_red, y_train, y_test = train_test_split(
    X_backward,
    y,
    test_size=0.2,
    random_state=42
)

model_reduced = LinearRegression()
model_reduced.fit(X_train_red, y_train)

y_pred_red = model_reduced.predict(X_test_red)

r2_reduced = r2_score(y_test, y_pred_red)


print("\n===== PORÓWNANIE =====")
print(f"R2 model pełny: {r2_full:.4f}")
print(f"R2 model zredukowany: {r2_reduced:.4f}")

print("\nPozostałe zmienne:")
print(X_backward.columns)