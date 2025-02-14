import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load("metrics.npy")

# Labels for the x-axis
methods = ["traj", "traj+2dbox"]
# methods = ["traj","traj+2dbox"] 
# Extracting columns
ade = data[:, 0]
fde = data[:, 1]
aswaee = data[:, 2]

# Plot
x = np.arange(len(methods))  # X-axis positions
width = 0.1  # Bar width

plt.figure(figsize=(10, 6))

# Creating bars
plt.bar(x - width, ade, width, label="ADE (m)")
plt.bar(x, fde, width, label="FDE (m)")
plt.bar(x + width, aswaee, width, label="ASWAEE (m)")

# Formatting the plot
plt.xticks(x, methods, rotation=25)
plt.ylabel("Error (m)")
plt.title("ADE, FDE, and ASWAEE Comparison on different evaluation modalities fro JRDB Dataset")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()
