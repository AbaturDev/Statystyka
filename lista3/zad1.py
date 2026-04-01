import numpy as np
from scipy import stats

np.random.seed(42)

kobiety = np.random.normal(loc=165, scale=6, size=30)
mezczyzni = np.random.normal(loc=178, scale=7, size=30)

t_stat, p_value = stats.ttest_ind(kobiety, mezczyzni)

print("Średni wzrost kobiet:", np.mean(kobiety))
print("Średni wzrost mężczyzn:", np.mean(mezczyzni))
print("Statystyka t:", t_stat)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Odrzucamy hipotezę zerową (istnieje istotna różnica).")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej.")