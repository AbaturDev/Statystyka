import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr, f_oneway

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "medical_data.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

df = df.dropna()

TARGET = "Cholesterol"

print(df["Smoker"].unique())

# 1. Czy palenie wpływa na poziom cholesterolu

smokers = df[
    df["Smoker"] == "Yes"
][TARGET]

non_smokers = df[
    df["Smoker"] == "No"
][TARGET]

print("\n=== ŚREDNI POZIOM CHOLESTEROLU WG PALENIA ===")
print(f"Palący: {smokers.mean():.2f}")
print(f"Niepalący: {non_smokers.mean():.2f}")

t_stat, p_value = ttest_ind(
    smokers,
    non_smokers
)

print("\n=== TEST T-STUDENTA ===")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna różnica.")
else:
    print("Brak statystycznie istotnej różnicy.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Smoker",
    y="Cholesterol"
)
plt.title("Poziom cholesterolu względem palenia")
plt.xlabel("Palenie papierosów")
plt.ylabel("Poziom cholesterolu")
plt.grid(True)
plt.savefig(os.path.join("images", "smoker_cholesterol_boxplot.png"))
plt.show()

# 2. Korelacja BMI i cholesterolu

correlation, p_value = pearsonr(
    df["BMI"],
    df["Cholesterol"]
)

print("\n=== KORELACJA BMI - CHOLESTEROL ===")
print(f"Korelacja Pearsona: {correlation:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna zależność.")
else:
    print("Brak statystycznie istotnej zależności.")

plt.figure(figsize=(8, 6))
plt.scatter(
    df["BMI"],
    df["Cholesterol"]
)
plt.xlabel("BMI")
plt.ylabel("Poziom cholesterolu")
plt.title("BMI a poziom cholesterolu")
plt.grid(True)
plt.savefig(os.path.join("images", "bmi_cholesterol.png"))
plt.show()

# 3. Poziom cholesterolu między grupami wiekowymi

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 50, 100],
    labels=["<30", "30-50", "50+"]
)

grouped = df.groupby(
    "AgeGroup"
)[TARGET]

print("\n=== ŚREDNI POZIOM CHOLESTEROLU WG WIEKU ===")
print(grouped.mean())

young = df[
    df["AgeGroup"] == "<30"
][TARGET]

middle = df[
    df["AgeGroup"] == "30-50"
][TARGET]

older = df[
    df["AgeGroup"] == "50+"
][TARGET]

anova_stat, anova_p = f_oneway(
    young,
    middle,
    older
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice między grupami wiekowymi.")
else:
    print("Brak statystycznie istotnych różnic między grupami wiekowymi.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="AgeGroup",
    y="Cholesterol"
)
plt.title("Poziom cholesterolu względem grup wiekowych")
plt.xlabel("Grupa wiekowa")
plt.ylabel("Poziom cholesterolu")
plt.grid(True)
plt.savefig(os.path.join("images", "agegroup_cholesterol_boxplot.png"))
plt.show()