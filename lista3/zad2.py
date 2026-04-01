import numpy as np
from scipy import stats

np.random.seed(42)

proba = np.random.normal(loc=168, scale=6, size=30)

t_stat, p_value = stats.ttest_1samp(proba, popmean=170)

print("Średnia z próby:", np.mean(proba))
print("Statystyka t:", t_stat)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Odrzucamy H0 – średnia różni się od 170 cm")
else:
    print("Brak podstaw do odrzucenia H0")