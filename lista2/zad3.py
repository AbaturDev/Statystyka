import numpy as np
from scipy.stats import ttest_ind

np.random.seed(42)
n = 50

sample1 = np.random.normal(loc=50, scale=10, size=n)
sample2 = np.random.normal(loc=55, scale=10, size=n)

t_stat, p_value = ttest_ind(sample1, sample2)

print(f"Średnia próby 1: {np.mean(sample1):.3f}")
print(f"Średnia próby 2: {np.mean(sample2):.3f}")
print(f"Statystyka t: {t_stat:.3f}")
print(f"Wartość p: {p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("Odrzucamy H0 – średnie różnią się istotnie")
else:
    print("Brak podstaw do odrzucenia H0 – średnie nie różnią się istotnie")