import numpy as np
from utils import stats, print_stats

np.random.seed(42)

population_size = 100000
mean = 50
std = 10

population = np.random.normal(mean, std, population_size)

sample_size = 5000
simple_random_sample = np.random.choice(population, sample_size)

median = np.median(population)
group1 = population[population <= median]
group2 = population[population > median]
stratified_sample = np.concatenate([
    np.random.choice(group1, sample_size // 2),
    np.random.choice(group2, sample_size // 2)
])

k = population_size // sample_size
systematic_sample = population[::k][:sample_size]

pop_mean, pop_std = stats(population)

rand_mean, rand_std = stats(simple_random_sample)
strat_mean, strat_std = stats(stratified_sample)
sys_mean, sys_std = stats(systematic_sample)

print_stats("Populacja:", pop_mean, pop_std)
print_stats("Próba losowa:", rand_mean, rand_std)
print_stats("Próba warstwowa:", strat_mean, strat_std)
print_stats("Próba systematyczna:", sys_mean, sys_std)