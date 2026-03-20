import numpy as np
from scipy.stats import chisquare

observed = np.array([18, 22, 20, 19, 21, 20])

expected = np.array([20, 20, 20, 20, 20, 20])

chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

print(f"Statystyka chi^2: {chi2_stat:.3f}")
print(f"Wartość p: {p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("Odrzucamy H0 – rozkład nie jest zgodny z oczekiwanym")
else:
    print("Brak podstaw do odrzucenia H0 – rozkład jest zgodny")