import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "online_shopping_data.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

df = df.dropna()

TARGET = "OrderValue"

# 1. Czy metoda płatności wpływa na średnią wartość koszyka

grouped = df.groupby(
    "PaymentMethod"
)[TARGET]

print("\n=== ŚREDNIA WARTOŚĆ KOSZYKA WG METODY PŁATNOŚCI ===")
print(grouped.mean())

payment_groups = [
    group[TARGET].values
    for _, group in df.groupby("PaymentMethod")
]

anova_stat, anova_p = f_oneway(
    *payment_groups
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice między metodami płatności.")
else:
    print("Brak statystycznie istotnych różnic między metodami płatności.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="PaymentMethod",
    y="OrderValue"
)
plt.title("Wartość koszyka względem metody płatności")
plt.xlabel("Metoda płatności")
plt.ylabel("Wartość zamówienia")
plt.grid(True)
plt.savefig(os.path.join("images", "payment_ordervalue_boxplot.png"))
plt.show()

# 2. Czy kategorie produktów mają różne średnie wartości zamówień

grouped = df.groupby(
    "Category"
)[TARGET]

print("\n=== ŚREDNIA WARTOŚĆ ZAMÓWIENIA WG KATEGORII ===")
print(grouped.mean())

category_groups = [
    group[TARGET].values
    for _, group in df.groupby("Category")
]

anova_stat, anova_p = f_oneway(
    *category_groups
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice między kategoriami produktów.")
else:
    print("Brak statystycznie istotnych różnic między kategoriami produktów.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Category",
    y="OrderValue"
)
plt.title("Wartość zamówienia względem kategorii produktu")
plt.xlabel("Kategoria produktu")
plt.ylabel("Wartość zamówienia")
plt.grid(True)
plt.savefig(os.path.join("images", "category_ordervalue_boxplot.png"))
plt.show()

# 3. Związek liczby produktów i wartości koszyka

correlation, p_value = pearsonr(
    df["ItemsInCart"],
    df["OrderValue"]
)

print("\n=== KORELACJA LICZBY PRODUKTÓW I WARTOŚCI KOSZYKA ===")
print(f"Korelacja Pearsona: {correlation:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna zależność.")
else:
    print("Brak statystycznie istotnej zależności.")

plt.figure(figsize=(8, 6))
plt.scatter(
    df["ItemsInCart"],
    df["OrderValue"]
)
plt.xlabel("Liczba produktów w koszyku")
plt.ylabel("Wartość koszyka")
plt.title("Liczba produktów a wartość koszyka")
plt.grid(True)
plt.savefig(os.path.join("images", "items_vs_ordervalue.png"))
plt.show()