import numpy as np
from scipy import stats

np.random.seed(42)

grupa1 = np.random.exponential(scale=1.0, size=30)
grupa2 = np.random.exponential(scale=1.5, size=30)

t_stat, p_t = stats.ttest_ind(grupa1, grupa2)

u_stat, p_mw = stats.mannwhitneyu(grupa1, grupa2)

print("Średnia grupa1:", np.mean(grupa1))
print("Średnia grupa2:", np.mean(grupa2))

print("\nTest t-Studenta:")
print("t =", t_stat)
print("p =", p_t)

print("\nTest Manna-Whitneya:")
print("U =", u_stat)
print("p =", p_mw)

alpha = 0.05

print("\nDecyzje:")
print("Test t:", "Odrzucamy H0" if p_t < alpha else "Brak podstaw")
print("Mann-Whitney:", "Odrzucamy H0" if p_mw < alpha else "Brak podstaw")