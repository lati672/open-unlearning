import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Generate two sample distributions
np.random.seed(42)
forget_scores = np.random.normal(0.5, 0.1, 100)  # Forgetting scores (mean=0.5, std=0.1)
retain_scores = np.random.normal(0.8, 0.1, 100)  # Retaining scores (mean=0.8, std=0.1)

# Compute empirical CDFs
forget_scores_sorted = np.sort(forget_scores)
retain_scores_sorted = np.sort(retain_scores)

forget_cdf = np.arange(1, len(forget_scores) + 1) / len(forget_scores)
retain_cdf = np.arange(1, len(retain_scores) + 1) / len(retain_scores)

# Perform KS test
ks_test = ks_2samp(forget_scores, retain_scores)
D_statistic = ks_test.statistic  # KS statistic
p_value = ks_test.pvalue  # p-value

# Plot the CDFs
plt.figure(figsize=(8, 5))
plt.plot(forget_scores_sorted, forget_cdf, label="Forget CDF", color="blue")
plt.plot(retain_scores_sorted, retain_cdf, label="Retain CDF", color="red")

# Find the max difference (D-statistic) location
x_D = forget_scores_sorted[np.argmax(np.abs(forget_cdf - retain_cdf))]
y_forget_D = forget_cdf[np.argmax(np.abs(forget_cdf - retain_cdf))]
y_retain_D = retain_cdf[np.argmax(np.abs(forget_cdf - retain_cdf))]

# Draw vertical line for KS statistic
plt.vlines(x_D, y_forget_D, y_retain_D, colors="black", linestyles="dashed", label=f"KS Statistic (D={D_statistic:.2f})")

plt.xlabel("Score")
plt.ylabel("Cumulative Probability")
plt.title("Kolmogorov-Smirnov (KS) Test Visualization")
plt.legend()
plt.grid(True)
plt.show(), (D_statistic, p_value)

