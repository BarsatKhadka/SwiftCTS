import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ----------------- DATA PREPARATION -----------------
raw_models = ["GCN", "SAGE", "GATv2"]
raw_vram = [1145.7, 717.6, 2499.6]
raw_time = [2264.3, 1801.4, 2106.4]

clustered_models = ["GCN", "SAGE", "GATv2"]
clustered_vram = [71.1, 39.3, 145.7]
clustered_time = [740.7, 604.8, 737.2]

# ----------------- MAIN PLOT SETUP -----------------
fig, ax = plt.subplots(figsize=(4, 3.2))

# Plot Raw Data
ax.scatter(raw_vram, raw_time, s=55, c='#1f77b4', marker='o', 
           label="Raw", edgecolor='k', linewidth=0.7, zorder=3)

# Plot Clustered Data
ax.scatter(clustered_vram, clustered_time, s=55, c='#ff7f0e', marker='^', 
           label="Clustered", edgecolor='k', linewidth=0.7, zorder=3)

# Manual Annotations for Main Plot
ax.annotate("GCN", (1145.7, 2264.3), xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight='bold')
ax.annotate("SAGE", (717.6, 1801.4), xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight='bold')
# Blue GATv2 annotation
ax.annotate("GATv2", (2499.6, 2106.4), xytext=(-38, 6), textcoords="offset points", fontsize=8, fontweight='bold')

# Styling Main Axes
ax.set_xlabel("Peak VRAM (MB)", fontsize=9, fontweight='bold')
ax.set_ylabel("Time (s)", fontsize=9, fontweight='bold')
ax.set_title("Efficiency Benchmark", fontsize=11, fontweight='bold', pad=15)

ax.set_xlim(-200, 2850) 
ax.set_ylim(400, 2600)

ax.tick_params(axis='both', which='major', labelsize=8)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc='upper left', fontsize=8, frameon=True)

# ----------------- INSET ZOOM SETUP -----------------
axins = inset_axes(ax, width="40%", height="38%", loc='lower right', 
                   bbox_to_anchor=(0, 0.08, 1, 1), bbox_transform=ax.transAxes)

axins.scatter(clustered_vram, clustered_time, s=45, c='#ff7f0e', marker='^', edgecolor='k', zorder=3)

# Ticks on top to prevent overlap
axins.xaxis.tick_top()
axins.xaxis.set_label_position('top')

# Inset Annotations
axins.annotate("SAGE", (39.3, 604.8), xytext=(6, -2), textcoords="offset points", fontsize=7, fontweight='bold')
axins.annotate("GCN", (71.1, 740.7), xytext=(6, 2), textcoords="offset points", fontsize=7, fontweight='bold')

# --- ORANGE GATV2 OFFSET ---
# xytext shifted to (-16, -12) as per your request
axins.annotate("GATv2", (145.7, 737.2), xytext=(-16, -12), textcoords="offset points", fontsize=7, fontweight='bold')

# Zoom Limits
axins.set_xlim(20, 180)
axins.set_ylim(550, 850)

axins.grid(True, linestyle="--", alpha=0.3)
axins.tick_params(axis='both', which='major', labelsize=7)
axins.text(0.7, 0.15, "Zoom", transform=axins.transAxes, 
           fontsize=8, fontweight='bold', color='gray', alpha=0.7)

# ----------------- CONNECTORS -----------------
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.savefig("efficiency_benchmark.pdf", format='pdf', bbox_inches='tight')
# ----------------- EXPORT -----------------
# dpi=300 ensures the image is high resolution for papers/presentations
plt.savefig("efficiency_benchmark.png", dpi=300, bbox_inches='tight')

plt.show()