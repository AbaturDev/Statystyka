import pandas as pd
from scipy.stats import shapiro, kstest, norm
import os

path = os.path.join("data", "SOCR-HeightWeight.csv")
data = pd.read_csv(path)
values = data["Height(Inches)"]

shapiro_stat, shapiro_p = shapiro(values)

print("Shapiro-Wilk:")
print(f"statystyka = {shapiro_stat:.3f}, p = {shapiro_p:.5f}")

mean = values.mean()
std = values.std()

ks_stat, ks_p = kstest(values, 'norm', args=(mean, std))

print("\nKołmogorow-Smirnow:")
print(f"statystyka = {ks_stat:.3f}, p = {ks_p:.5f}")

alpha = 0.05

if shapiro_p < alpha:
    print("\nShapiro: dane NIE są normalne")
else:
    print("\nShapiro: brak podstaw do odrzucenia normalności")

if ks_p < alpha:
    print("KS: dane NIE są normalne")
else:
    print("KS: brak podstaw do odrzucenia normalności")