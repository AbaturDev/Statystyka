import numpy as np
from scipy.stats import ttest_1samp

np.random.seed(42)
mu_true = 52
n = 30
mu_0 = 50
alpha = 0.05

sample = np.random.normal(loc=mu_true, scale=10, size=n)

t_stat, p_value = ttest_1samp(sample, mu_0)

print(f"Średnia z próby: {np.mean(sample):.3f}")
print(f"Statystyka t: {t_stat:.3f}")
print(f"Wartość p: {p_value:.5f}")

if p_value < alpha:
    print("Odrzucamy H0 – średnia różni się istotnie od 50")
else:
    print("Brak podstaw do odrzucenia H0 – średnia nie różni się istotnie od 50")