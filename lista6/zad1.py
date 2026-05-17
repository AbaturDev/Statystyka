import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, pearsonr, f_oneway
from statsmodels.stats.anova import anova_lm

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "student_math_scores.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

df = df.dropna()

TARGET = "MathScore"

# 1. Czy średni wynik różni się między płciami

male_scores = df[
    df["Gender"] == "Male"
][TARGET]

female_scores = df[
    df["Gender"] == "Female"
][TARGET]

print("\n=== ŚREDNI WYNIK WG PŁCI ===")
print(f"Mężczyźni: {male_scores.mean():.2f}")
print(f"Kobiety: {female_scores.mean():.2f}")

t_stat, p_value = ttest_ind(
    male_scores,
    female_scores
)

print("\n=== TEST T-STUDENTA ===")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna różnica między płciami.")
else:
    print("Brak statystycznie istotnej różnicy między płciami.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Gender",
    y="MathScore"
)
plt.title("Wyniki matematyki względem płci")
plt.xlabel("Płeć")
plt.ylabel("Wynik z matematyki")
plt.grid(True)
plt.savefig(os.path.join("images", "gender_mathscore_boxplot.png"))
plt.show()

# 2. Zależność godzin nauki i wyniku

correlation, p_value = pearsonr(
    df["StudyHours"],
    df["MathScore"]
)

print("\n=== KORELACJA GODZINY NAUKI - WYNIK ===")
print(f"Korelacja Pearsona: {correlation:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna zależność.")
else:
    print("Brak statystycznie istotnej zależności.")

plt.figure(figsize=(8, 6))
plt.scatter(
    df["StudyHours"],
    df["MathScore"]
)
plt.xlabel("Godziny nauki tygodniowo")
plt.ylabel("Wynik z matematyki")
plt.title("Godziny nauki a wynik z matematyki")
plt.grid(True)
plt.savefig(os.path.join( "images", "studyhours_vs_mathscore.png" ))
plt.show()

# 3. Status socjoekonomiczny a wynik

grouped = df.groupby(
    "SocioeconomicStatus"
)[TARGET]

print("\n=== ŚREDNIE WG STATUSU SOCJOEKONOMICZNEGO ===")
print(grouped.mean())

low_scores = df[
    df["SocioeconomicStatus"] == "Low"
][TARGET]

medium_scores = df[
    df["SocioeconomicStatus"] == "Medium"
][TARGET]

high_scores = df[
    df["SocioeconomicStatus"] == "High"
][TARGET]

anova_stat, anova_p = f_oneway(
    low_scores,
    medium_scores,
    high_scores
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice między grupami.")
else:
    print("Brak statystycznie istotnych różnic między grupami.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="SocioeconomicStatus",
    y="MathScore"
)
plt.title("Wyniki matematyki względem statusu socjoekonomicznego")
plt.xlabel("Status socjoekonomiczny")
plt.ylabel("Wynik z matematyki")
plt.grid(True)
plt.savefig(os.path.join("images","socioeconomic_mathscore_boxplot.png"))
plt.show()

print("\n=== DWUCZYNNIKOWA ANOVA ===")

anova_model = smf.ols(
    "MathScore ~ C(Gender) + C(SocioeconomicStatus) + C(Gender):C(SocioeconomicStatus)",
    data=df
).fit()

anova_results = anova_lm(anova_model)

print(anova_results)