import numpy as np
from utils import stats, print_stats

np.random.seed(42)

population_size = 100000
sample_size = 5000

group1 = np.random.normal(40, 5, population_size // 2)
group2 = np.random.normal(70, 8, population_size // 2)

population = np.concatenate([group1, group2])
random_sample = np.random.choice(population, sample_size)
biased_sample = np.random.choice(group1, sample_size)

pop_mean, pop_std = stats(population)

rand_mean, rand_std = stats(random_sample)

bias_mean, bias_std = stats(biased_sample)

print_stats("Populacja:", pop_mean, pop_std)
print_stats(f"Próba losowa:", rand_mean, rand_std)
print_stats(f"Próba nierandomizowana:", bias_mean, bias_std)
