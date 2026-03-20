import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

np.random.seed(42)
n = 40

group1 = np.random.normal(loc=50, scale=5, size=n)
group2 = np.random.normal(loc=55, scale=5, size=n)
group3 = np.random.normal(loc=60, scale=5, size=n)

f_stat, p_value = f_oneway(group1, group2, group3)

print(f"Statystyka F: {f_stat:.3f}")
print(f"Wartość p: {p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("Odrzucamy H0 – przynajmniej jedna średnia się różni")

    data = np.concatenate([group1, group2, group3])
    labels = (["G1"] * n) + (["G2"] * n) + (["G3"] * n)

    tukey = pairwise_tukeyhsd(data, labels)
    print("\nTest post hoc (Tukey):")
    print(tukey)

else:
    print("Brak podstaw do odrzucenia H0 – wszystkie średnie są równe")