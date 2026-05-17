import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "ad_campaign_data.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

df = df.dropna()

TARGET = "Clicks"

# 1. Skuteczność różnych typów reklam

grouped = df.groupby(
    "AdType"
)[TARGET]

print("\n=== ŚREDNIA LICZBA KLIKNIĘĆ WG TYPU REKLAMY ===")
print(grouped.mean())

ad_groups = [
    group[TARGET].values
    for _, group in df.groupby("AdType")
]

anova_stat, anova_p = f_oneway(
    *ad_groups
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice między typami reklam.")
else:
    print("Brak statystycznie istotnych różnic między typami reklam.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="AdType",
    y="Clicks"
)
plt.title("Liczba kliknięć względem typu reklamy")
plt.xlabel("Typ reklamy")
plt.ylabel("Liczba kliknięć")
plt.grid(True)
plt.savefig(os.path.join("images", "adtype_clicks_boxplot.png"))
plt.show()

# 2. Pora dnia a liczba kliknięć

grouped = df.groupby(
    "TimeOfDay"
)[TARGET]

print("\n=== ŚREDNIA LICZBA KLIKNIĘĆ WG PORY DNIA ===")
print(grouped.mean())

time_groups = [
    group[TARGET].values
    for _, group in df.groupby("TimeOfDay")
]

anova_stat, anova_p = f_oneway(
    *time_groups
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice między porami dnia.")
else:
    print("Brak statystycznie istotnych różnic między porami dnia.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="TimeOfDay",
    y="Clicks"
)
plt.title("Liczba kliknięć względem pory dnia")
plt.xlabel("Pora dnia")
plt.ylabel("Liczba kliknięć")
plt.grid(True)
plt.savefig(os.path.join("images", "timeofday_clicks_boxplot.png"))
plt.show()

# 3. Typ reklamy a wskaźnik konwersji

contingency_table = pd.crosstab(
    df["AdType"],
    df["Conversion"]
)

print("\n=== TABELA KONTYNGENCJI ===")
print(contingency_table)

chi2, p_value, dof, expected = chi2_contingency(
    contingency_table
)

print("\n=== TEST CHI-KWADRAT ===")
print(f"Chi2 statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna zależność między typem reklamy a konwersją.")
else:
    print("Brak statystycznie istotnej zależności między typem reklamy a konwersją.")

plt.figure(figsize=(8, 6))
sns.heatmap(
    contingency_table,
    annot=True,
    cmap="Blues",
    fmt="d"
)
plt.title("Typ reklamy a konwersja")
plt.xlabel("Konwersja")
plt.ylabel("Typ reklamy")
plt.savefig(os.path.join("images", "adtype_conversion_heatmap.png"))
plt.show()