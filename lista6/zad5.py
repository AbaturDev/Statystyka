import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, ttest_ind
from statsmodels.stats.anova import anova_lm

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "life_satisfaction_activity.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

df = df.dropna()

TARGET = "LifeSatisfaction"

# 1. Korelacja między zadowoleniem z życia a aktywnością fizyczną

correlation, p_value = pearsonr(
    df["ActivityDays"],
    df["LifeSatisfaction"]
)

print("\n=== KORELACJA AKTYWNOŚĆ - ZADOWOLENIE Z ŻYCIA ===")
print(f"Korelacja Pearsona: {correlation:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna zależność.")
else:
    print("Brak statystycznie istotnej zależności.")

plt.figure(figsize=(8, 6))
plt.scatter(
    df["ActivityDays"],
    df["LifeSatisfaction"]
)
plt.xlabel("Liczba dni aktywności fizycznej")
plt.ylabel("Zadowolenie z życia")
plt.title("Aktywność fizyczna a zadowolenie z życia")
plt.grid(True)
plt.savefig(os.path.join("images", "activity_vs_lifesatisfaction.png"))
plt.show()

# 2. Porównanie pracujących i bezrobotnych

employed_scores = df[
    df["EmploymentStatus"] == "Employed"
][TARGET]

unemployed_scores = df[
    df["EmploymentStatus"] == "Unemployed"
][TARGET]

print("\n=== ŚREDNIE ZADOWOLENIE WG STATUSU ZATRUDNIENIA ===")
print(f"Pracujący: {employed_scores.mean():.2f}")
print(f"Bezrobotni: {unemployed_scores.mean():.2f}")

t_stat, p_value = ttest_ind(
    employed_scores,
    unemployed_scores
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
    x="EmploymentStatus",
    y="LifeSatisfaction"
)
plt.title("Zadowolenie z życia względem statusu zatrudnienia")
plt.xlabel("Status zatrudnienia")
plt.ylabel("Zadowolenie z życia")
plt.grid(True)
plt.savefig(os.path.join("images", "employment_lifesatisfaction_boxplot.png"))
plt.show()

# 3. Czy aktywność wpływa inaczej na kobiety i mężczyzn

print("\n=== ANALIZA INTERAKCJI PŁEĆ × AKTYWNOŚĆ ===")

anova_model = smf.ols(
    "LifeSatisfaction ~ C(Gender) + ActivityDays + C(Gender):ActivityDays",
    data=df
).fit()

anova_results = anova_lm(anova_model)

print(anova_results)