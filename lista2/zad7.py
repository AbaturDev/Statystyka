import numpy as np
from scipy.stats import pearsonr, spearmanr

np.random.seed(42)
n = 50

x = np.random.normal(0, 1, n)
y_corr = 2 * x + np.random.normal(0, 0.5, n)

y_uncorr = np.random.normal(0, 1, n)

pearson_corr, pearson_p = pearsonr(x, y_corr)

spearman_corr, spearman_p = spearmanr(x, y_corr)

print("Dane skorelowane")
print(f"Pearson r = {pearson_corr:.3f}, p = {pearson_p:.5f}")
print(f"Spearman rho = {spearman_corr:.3f}, p = {spearman_p:.5f}")

pearson_corr2, pearson_p2 = pearsonr(x, y_uncorr)
spearman_corr2, spearman_p2 = spearmanr(x, y_uncorr)

print("\nDane nieskorelowane")
print(f"Pearson r = {pearson_corr2:.3f}, p = {pearson_p2:.5f}")
print(f"Spearman rho = {spearman_corr2:.3f}, p = {spearman_p2:.5f}")

alpha = 0.05

print("\nInterpretacja")
if pearson_p < alpha:
    print("Dane skorelowane: istotna korelacja (Pearson)")
else:
    print("Dane skorelowane: brak istotnej korelacji (Pearson)")

if pearson_p2 < alpha:
    print("Dane nieskorelowane: istotna korelacja (Pearson)")
else:
    print("Dane nieskorelowane: brak istotnej korelacji (Pearson)")