import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cte import sigma_eta, sigma_u, N
# Set random seed for reproducibility
np.random.seed(42)

# Number of samples

# Sample from prior: u ~ N(0, 1)
u = np.random.normal(loc=0, scale=sigma_eta, size=N)

# Sample noise: eta ~ N(0, 1)
eta = np.random.normal(loc=0, scale=sigma_u, size=N)

# Forward model: y = u^2 + eta
y = u**2 + eta

print("Samples generated")
# Save samples
np.save("./../../Data/Experiment1/samples_u.npy", u)
np.save("./../../Data/Experiment1/samples_y.npy", y)
print("Samples Saved")

# Plot prior distribution of u
plt.figure(figsize=(6, 4))
sns.histplot(u, bins=50, kde=True, stat="density", color="skyblue", edgecolor="black")
plt.title("Prior Distribution of $u$")
plt.xlabel("$u$")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.savefig("./../../Figs/Experiment1/prior_u.png")
plt.show()

# Plot joint distribution using hist2d for better control and colorbar
plt.figure(figsize=(6, 5))
hb = plt.hist2d(u, y, bins=100, cmap='viridis', density=True)
plt.colorbar(hb[3], label=r"$\mathbb{P}(u, y)$")
plt.xlabel("$u$")
plt.ylabel("$y$")
# plt.grid(True)
plt.tight_layout()
plt.savefig("./../../Figs/Experiment1/joint_distribution_with_colorbar.png",dpi=300)
plt.show()
