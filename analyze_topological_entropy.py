import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the CSV file
df = pd.read_csv('eigenvalues_tokens.csv')  # Replace with your actual file

# Extract eigenvalues and filter out non-positive values
eigenvalues = df['eigenvalue'].values
eigenvalues = eigenvalues[eigenvalues > 0]

# Histogram the eigenvalues
counts, bin_edges = np.histogram(eigenvalues, bins=100)

# Compute bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Remove bins with zero count (to avoid log(0))
mask = counts > 0
log_bin_centers = np.log10(bin_centers[mask])
log_counts = np.log10(counts[mask])

# Linear regression (slope of the log-log plot)
slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers, log_counts)

# Plot log-log distribution
plt.figure(figsize=(8, 6))
plt.plot(log_bin_centers, log_counts, marker='o', linestyle='', alpha=0.7, label='Data')
plt.plot(log_bin_centers, slope * log_bin_centers + intercept, color='red', linestyle='--',
         label=f'Fit: slope = {slope:.2f}')
plt.xlabel(r'$\log_{10}(\lambda)$', fontsize=12)
plt.ylabel(r'$\log_{10}(\mathrm{frequency})$', fontsize=12)
plt.title('Log-Log Distribution of Eigenvalues', fontsize=14)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope: {slope:.4f}, R-squared: {r_value**2:.4f}")
