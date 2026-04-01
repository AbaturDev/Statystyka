import numpy as np
from scipy.stats import chi2_contingency

# Tabela kontyngencji (wiersze: płeć, kolumny: gatunek muzyki)
dane = np.array([
    [20, 5, 5],   # kobiety
    [10, 15, 5]   # mężczyźni
])

chi2, p_value, dof, expected = chi2_contingency(dane)

print("Statystyka chi2:", chi2)
print("p-value:", p_value)
print("Stopnie swobody:", dof)
print("Wartości oczekiwane:\n", expected)

alpha = 0.05
if p_value < alpha:
    print("Odrzucamy H0 – istnieje zależność między płcią a preferencjami muzycznymi")
else:
    print("Brak podstaw do odrzucenia H0 – brak zależności")