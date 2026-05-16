import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

path = os.path.join("data", "mpg.csv")
df = pd.read_csv(path)

df = df.dropna()

TARGET = "mpg"

df = pd.get_dummies(
    df,
    columns=["origin"],
    drop_first=True
)

model_features = {

    "Pełny": [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin_japan",
        "origin_usa"
    ],

    "Po eliminacji": [
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin_japan",
        "origin_usa"
    ],

    "Tylko twarde dane": [
        "cylinders",
        "horsepower",
        "weight",
        "displacement"
    ]
}

results = []

for model_name, features in model_features.items():

    X = df[features]
    y = df[TARGET]

    # =========================
    # TRAIN / TEST
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    # =========================
    # VIF
    # =========================

    X_vif = add_constant(X)
    X_vif = X_vif.astype(float)

    vif_values = []

    for i in range(X_vif.shape[1]):

        vif = variance_inflation_factor(
            X_vif.values,
            i
        )

        vif_values.append(vif)

    vif_df = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": vif_values
    })

    vif_df = vif_df[
        vif_df["Feature"] != "const"
    ]

    avg_vif = vif_df["VIF"].mean()

    results.append({
        "Model": model_name,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "Średni VIF": round(avg_vif, 4)
    })

    print(f"\n===== {model_name.upper()} =====")

    print("R2:", round(r2, 4))
    print("RMSE:", round(rmse, 4))

    print("\nVIF:")
    print(vif_df)

results_df = pd.DataFrame(results)

print("\n===== PORÓWNANIE MODELI =====")
print(results_df)

best_model = results_df.sort_values(
    by=["Średni VIF", "R2", "RMSE"],
    ascending=[True, False, True]
).iloc[0]

print("\n===== NAJLEPSZY MODEL =====")
print(best_model)