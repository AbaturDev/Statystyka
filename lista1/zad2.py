import numpy as np
from scipy import stats

np.random.seed(42)

mean = 100
std = 15
population_size = 100000

population = np.random.normal(mean, std, population_size)
sample_sizes = [10, 50, 1000, 5000, 10000, 50000]

def confidence_interval(sample):
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)

    z = 1.96 # 95%
    margin = z * (std / np.sqrt(n))

    lower = mean - margin
    upper = mean + margin

    return mean, lower, upper


for size in sample_sizes:
    sample = np.random.choice(population, size)
    mean, lower, upper = confidence_interval(sample)

    print(f"\nPróba n={size}")
    print(f"Średnia: {mean}")
    print(f"95% CI: ({lower}, {upper})")
    print(f"Szerokość przedziału: {upper-lower}")