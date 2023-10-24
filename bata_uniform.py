import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import concurrent.futures
import os

def compute_second_largest(args):
    num_samples, n = args
    samples = np.random.uniform(0, 1, (num_samples, n))
    second_largest = np.partition(samples, -2)[:, -2]
    return second_largest

if __name__ == "__main__":
    n = 1000
    num_samples = 100000
    num_processes = os.cpu_count()
    samples_per_process = num_samples // num_processes

    args = [(samples_per_process, n) for _ in range(num_processes)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(compute_second_largest, args))

    second_largest_samples = np.concatenate(results)

    counts, bins, _ = plt.hist(second_largest_samples, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    a = n-1
    b = 2

    y = beta.pdf(bin_centers, a, b)

    plt.plot(bin_centers, y, 'r-', label=f'Beta({a},{b})')
    plt.title('Distribution of the Second Largest Value')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
