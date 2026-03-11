import numpy as np
from utils import stats, print_stats

np.random.seed(42)

mean = 50
std = 10
population_size = 100000
sample_sizes = [10, 50, 1000, 5000, 10000, 50000]

population = np.random.normal(mean, std, population_size)

pop_mean, pop_std = stats(population)

print_stats("Populacja:", pop_mean, pop_std)

for size in sample_sizes:
    sample = np.random.choice(population, size)
    mean, std = stats(sample)

    print_stats(f"Próba {size}:", mean, std)
