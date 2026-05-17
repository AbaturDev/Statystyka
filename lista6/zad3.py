import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "movie_preferences.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

df = df.dropna()

TARGET = "Rating"

# 1. Czy grupy wiekowe preferują różne gatunki filmowe

contingency_table = pd.crosstab(
    df["AgeGroup"],
    df["Genre"]
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
    print("Istnieje statystycznie istotna zależność między grupą wiekową a gatunkiem filmu.")
else:
    print("Brak statystycznie istotnej zależności między grupą wiekową a gatunkiem filmu.")

plt.figure(figsize=(8, 6))
sns.heatmap(
    contingency_table,
    annot=True,
    cmap="Blues",
    fmt="d"
)
plt.title("Preferencje gatunków filmowych względem grup wiekowych")
plt.xlabel("Gatunek filmu")
plt.ylabel("Grupa wiekowa")
plt.savefig(os.path.join("images", "agegroup_genre_heatmap.png"))
plt.show()

# 2. Czy oceny filmów różnią się między płciami

male_ratings = df[
    df["Gender"] == "Male"
][TARGET]

female_ratings = df[
    df["Gender"] == "Female"
][TARGET]

print("\n=== ŚREDNIE OCENY WG PŁCI ===")
print(f"Mężczyźni: {male_ratings.mean():.2f}")
print(f"Kobiety: {female_ratings.mean():.2f}")

t_stat, p_value = ttest_ind(
    male_ratings,
    female_ratings
)

print("\n=== TEST T-STUDENTA ===")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna różnica ocen między płciami.")
else:
    print("Brak statystycznie istotnej różnicy ocen między płciami.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Gender",
    y="Rating"
)
plt.title("Oceny filmów względem płci")
plt.xlabel("Płeć")
plt.ylabel("Ocena filmu")
plt.grid(True)
plt.savefig(os.path.join("images", "gender_rating_boxplot.png"))
plt.show()

# 3. Związek między wiekiem a oceną filmu

grouped = df.groupby("AgeGroup")[TARGET]

print("\n=== ŚREDNIE OCENY WG GRUPY WIEKOWEJ ===")
print(grouped.mean())

young_ratings = df[
    df["AgeGroup"] == "18-25"
][TARGET]

middle_ratings = df[
    df["AgeGroup"] == "26-40"
][TARGET]

older_ratings = df[
    df["AgeGroup"] == "41+"
][TARGET]

under18_ratings = df[
    df["AgeGroup"] == "<18"
][TARGET]

anova_stat, anova_p = f_oneway(
    young_ratings,
    middle_ratings,
    older_ratings,
    under18_ratings
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice ocen między grupami wiekowymi.")
else:
    print("Brak statystycznie istotnych różnic ocen między grupami wiekowymi.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="AgeGroup",
    y="Rating"
)
plt.title("Oceny filmów względem grup wiekowych")
plt.xlabel("Grupa wiekowa")
plt.ylabel("Ocena filmu")
plt.grid(True)
plt.savefig(os.path.join("images", "agegroup_rating_boxplot.png"))
plt.show()