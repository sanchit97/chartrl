# import matplotlib.pyplot as plt
# import numpy as np

# # Same 7-pt size for labels, ticks, legend, etc.
# plt.rcParams.update({
#     "font.size": 7,
#     "axes.titlesize": 7,
#     "axes.labelsize": 7,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
#     "legend.fontsize": 7,
# })

# # Data
# psi       = np.array([1e-1, 1e0, 1e1, 1e2])
# accuracy  = np.array([84.6, 85.7, 85.1, 83.9])
# baseline  = 84.5

# # Figure
# fig, ax = plt.subplots(figsize=(3, 2))          # fixed size in inches
# ax.plot(psi, accuracy, marker="D", linewidth=0.8)
# ax.axhline(baseline, linestyle="--", linewidth=0.8)

# ax.set_xscale("log")
# ax.set_xlabel(r"$\Psi$")
# ax.set_ylabel("Accuracy")

# fig.tight_layout()
# plt.show()

# fig.savefig("psi_ablation_7pt.pdf")   # vector output for LaTeX
# # plt.savefig("psi_ablation_7pt.png", dpi=300)  # raster output for other uses

import matplotlib.pyplot as plt
import numpy as np

# --- Global 7-pt font -------------------------------------------------
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

# --- Data -------------------------------------------------------------
heads     = np.array([4, 8, 16, 32])
accuracy  = np.array([82.6, 85.1, 85.7, 85.6])
baseline  = 84.5

# --- Plot -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 2))      # fixed canvas
ax.plot(heads, accuracy, marker="D", linewidth=0.8)
ax.axhline(baseline, linestyle="--", linewidth=0.8, color="red")

ax.set_xlabel("Number of heads in MSFI")
ax.set_ylabel("Accuracy")

# --- Remove outer whitespace -----------------------------------------

plt.show()
fig.savefig("heads_ablation_7pt.pdf")  # vector output for LaTeX
