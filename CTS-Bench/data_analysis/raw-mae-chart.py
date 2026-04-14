import matplotlib.pyplot as plt
import numpy as np

# ----------------- DATA PREPARATION -----------------
nets = ["GCN", "SAGE", "GATv2"]
metrics = ["Skew MAE", "Power MAE", "Wire MAE"]

mae_seen = [[0.16, 0.16, 0.16], [0.06, 0.06, 0.06], [0.06, 0.06, 0.06]]
mae_unseen = [[0.24, 0.26, 0.22], [0.07, 0.06, 0.06], [0.11, 0.11, 0.11]]

# Spatial R2 values from your summary table
r2_seen = [0.9, 0.92, 0.88]
r2_unseen = [-0.17, 0.05, 0.08]

# ----------------- PLOT SETUP -----------------
# We use a taller figure for the 2x2 stack
fig = plt.figure(figsize=(8, 7))
x = np.arange(len(nets))
width = 0.35 

# Creating the grid: (2 rows, 2 columns)
# Skew (0,0), Power (0,1), Wire (1,0)
ax_skew = plt.subplot2grid((2, 2), (0, 0))
ax_power = plt.subplot2grid((2, 2), (0, 1))
ax_wire = plt.subplot2grid((2, 2), (1, 0))
axes = [ax_skew, ax_power, ax_wire]

for i, ax in enumerate(axes):
    # 1. Plot MAE Bars
    b1 = ax.bar(x - width/2, mae_seen[i], width, label='MAE Seen', color='#1f77b4', edgecolor='k', alpha=0.8, zorder=3)
    b2 = ax.bar(x + width/2, mae_unseen[i], width, label='MAE Unseen', color='#ff7f0e', edgecolor='k', alpha=0.8, zorder=3)
    
    # Auto-scaling Y to prevent number/border collision
    ax.set_ylim(0, max(max(mae_unseen[i]), max(mae_seen[i])) * 1.4) 
    
    # 2. Add Bar Labels (with nudge for identical values)
    for rect_s, rect_u in zip(b1, b2):
        h_s, h_u = rect_s.get_height(), rect_u.get_height()
        off_s, off_u = (0, 0)
        if abs(h_s - h_u) < 0.01: off_s, off_u = (-5, 5) # Slight nudge

        ax.annotate(f'{h_s:.2f}', xy=(rect_s.get_x() + width/2, h_s),
                    xytext=(off_s, 6), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.annotate(f'{h_u:.2f}', xy=(rect_u.get_x() + width/2, h_u),
                    xytext=(off_u, 6), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Plot R2 (Secondary Axis)
    ax2 = ax.twinx()
    ln1 = ax2.plot(x - width/2, r2_seen, 'D', color='red', markersize=6, label='$R^2$ Seen', markeredgecolor='k', zorder=5)
    ln2 = ax2.plot(x + width/2, r2_unseen, 's', color='darkred', markersize=6, label='$R^2$ Unseen', markeredgecolor='k', zorder=5)
    
    ax2.set_ylim(-0.5, 1.2)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=9)
    
    # Labels and Titles
    ax.set_title(metrics[i], fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(nets, fontsize=10)
    ax.set_ylabel('MAE', fontsize=10, fontweight='bold')
    if i == 1 or i == 2: # Add R2 label to the right-side charts
        ax2.set_ylabel('Spatial $R^2$', fontsize=10, fontweight='bold', color='red')

# --- 4. LEGEND IN THE EMPTY QUADRANT (1,1) ---
ax_leg = plt.subplot2grid((2, 2), (1, 1))
ax_leg.axis('off') # Hide the axis for the legend box

# Get labels from one of the plots
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

# Vertical legend in the center of the empty space
ax_leg.legend(h1 + h2, l1 + l2, loc='center', fontsize=11, frameon=True, borderpad=1.5, labelspacing=1.2)

# ----------------- EXPORT -----------------
plt.tight_layout()
plt.savefig("mae_r2_grid_layout.pdf", format='pdf', bbox_inches='tight')
plt.savefig("mae_r2_grid_layout.png", dpi=300, bbox_inches='tight')
plt.show()