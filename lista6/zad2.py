import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway

os.makedirs("images", exist_ok=True)

path = os.path.join("data", "coffee_stress_data.csv")
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

df = df.dropna()

# 1. Czy liczba filiżanek kawy koreluje z poziomem stresu

correlation, p_value = pearsonr(
    df["CoffeeCupsPerDay"],
    df["StressLevel"]
)

print("\n=== KORELACJA KAWA - STRES ===")
print(f"Korelacja Pearsona: {correlation:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Istnieje statystycznie istotna zależność.")
else:
    print("Brak statystycznie istotnej zależności.")

plt.figure(figsize=(8, 6))
plt.scatter(
    df["CoffeeCupsPerDay"],
    df["StressLevel"]
)
plt.xlabel("Liczba filiżanek kawy dziennie")
plt.ylabel("Poziom stresu")
plt.title("Konsumpcja kawy a poziom stresu")
plt.grid(True)
plt.savefig(os.path.join("images", "coffee_vs_stress.png"))
plt.show()

# 2. Porównanie poziomu stresu między środowiskami pracy

grouped = df.groupby("Workplace")["StressLevel"]

print("\n=== ŚREDNI POZIOM STRESU WG ŚRODOWISKA PRACY ===")
print(grouped.mean())

office_stress = df[
    df["Workplace"] == "Office"
]["StressLevel"]

home_stress = df[
    df["Workplace"] == "Home"
]["StressLevel"]

other_stress = df[
    df["Workplace"] == "Other"
]["StressLevel"]

anova_stat, anova_p = f_oneway(
    office_stress,
    home_stress,
    other_stress
)

print("\n=== TEST ANOVA ===")
print(f"F-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < 0.05:
    print("Istnieją statystycznie istotne różnice między środowiskami pracy.")
else:
    print("Brak statystycznie istotnych różnic między środowiskami pracy.")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Workplace",
    y="StressLevel"
)
plt.title("Poziom stresu względem środowiska pracy")
plt.xlabel("Środowisko pracy")
plt.ylabel("Poziom stresu")
plt.grid(True)
plt.savefig(os.path.join("images", "stress_by_workplace.png"))
plt.show()

# 3. Czy wiek ma wpływ na spożycie kawy

correlation_age, p_value_age = pearsonr(
    df["Age"],
    df["CoffeeCupsPerDay"]
)

print("\n=== KORELACJA WIEK - SPOŻYCIE KAWY ===")
print(f"Korelacja Pearsona: {correlation_age:.4f}")
print(f"p-value: {p_value_age:.4f}")

if p_value_age < 0.05:
    print("Wiek ma statystycznie istotny wpływ na spożycie kawy.")
else:
    print("Brak statystycznie istotnego wpływu wieku na spożycie kawy.")

plt.figure(figsize=(8, 6))
plt.scatter(
    df["Age"],
    df["CoffeeCupsPerDay"]
)
plt.xlabel("Wiek")
plt.ylabel("Liczba filiżanek kawy dziennie")
plt.title("Wiek a spożycie kawy")
plt.grid(True)
plt.savefig(os.path.join("images", "age_vs_coffee.png"))
plt.show()