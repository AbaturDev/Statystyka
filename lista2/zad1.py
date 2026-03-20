import numpy as np
from scipy.stats import norm

np.random.seed(42)
mu_true = 52
sigma = 5
n = 100
mu_0 = 50
alpha = 0.05

sample = np.random.normal(loc=mu_true, scale=sigma, size=n)

x_bar = np.mean(sample)

Z = (x_bar - mu_0) / (sigma / np.sqrt(n))

p_value = 2 * (1 - norm.cdf(abs(Z)))

print(f"Średnia z próby: {x_bar:.3f}")
print(f"Statystyka Z: {Z:.3f}")
print(f"Wartość p: {p_value:.5f}")

if p_value < alpha:
    print("Odrzucamy H0 – średnia różni się istotnie od 50")
else:
    print("Brak podstaw do odrzucenia H0 – średnia nie różni się istotnie od 50")