import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


os.makedirs("images", exist_ok=True)

np.random.seed(42)

sample_sizes = np.arange(10, 1001, 10)

p_values = []

for n in sample_sizes:
    group1 = np.random.normal(loc=170, scale=6, size=n)
    group2 = np.random.normal(loc=171, scale=6, size=n)
    
    t_stat, p = stats.ttest_ind(group1, group2)
    p_values.append(p)

plt.figure()
plt.plot(sample_sizes, p_values)
plt.axhline(y=0.05)
plt.xlabel("Liczność próby (n)")
plt.ylabel("p-value")
plt.title("Wpływ liczności próby na wartość p")
plt.savefig(os.path.join("images", "zad5.png"))
plt.show()