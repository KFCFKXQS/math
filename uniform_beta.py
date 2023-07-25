# digression
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

n = 1000
num_samples = 100000

samples = np.random.uniform(0, 1, (num_samples, n))
sorted_samples = np.sort(samples, axis=1)
second_largest_values = sorted_samples[:, -2]



counts, bins = np.histogram(second_largest_values, bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

a = n-1
b = 2

y = beta.pdf(bin_centers, a, b)

plt.hist(second_largest_values, bins=50, density=True, alpha=0.6, color='g', label='Simulation')
plt.plot(bin_centers, y, 'r-', label=f'Beta({a},{b})')
plt.title('Distribution of the Second Largest Value')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
