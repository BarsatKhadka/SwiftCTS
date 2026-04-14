import matplotlib.pyplot as plt
import numpy as np

# ----------------- DATA PREPARATION (CLUSTERED) -----------------
nets = ["GCN", "SAGE", "GATv2"]
metrics = ["Skew MAE", "Power MAE", "Wire MAE"]

mae_seen = [[0.17, 0.17, 0.17], [0.10, 0.09, 0.09], [0.10, 0.09, 0.09]]
mae_unseen = [[0.16, 0.12, 0.06], [0.13, 0.09, 0.13], [0.15, 0.16, 0.23]]
r2_seen = [0.85, 0.90, 0.89]
r2_unseen = [-2.23, -0.91, -2.84]

# Function to detect overlap and move labels inside bars if needed
def smart_annotate(ax, rect, mae, r2, mae_ylim, r2_ylim):
    # Map R2 value to MAE axis space to check proximity
    r2_range = r2_ylim[1] - r2_ylim[0]
    mae_range = mae_ylim[1] - 0
    rel_height = (r2 - r2_ylim[0]) / r2_range
    r2_on_mae = rel_height * mae_range
    
    # Threshold for overlap (approx. height of text labels)
    threshold = 0.12 * mae_range
    
    if abs(r2_on_mae - mae) < threshold:
        # OVERLAP DETECTED: Write number INSIDE the bar
        color = 'white' if rect.get_facecolor() == '#1f77b4' else 'black'
        ax.annotate(f'{mae:.2f}', xy=(rect.get_x() + rect.get_width()/2, mae),
                    xytext=(0, -12), textcoords="offset points", ha='center', va='top', 
                    fontsize=9, fontweight='bold', color=color)
    else:
        # NO OVERLAP: Write number ABOVE the bar
        ax.annotate(f'{mae:.2f}', xy=(rect.get_x() + rect.get_width()/2, mae),
                    xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')

# ----------------- PLOT SETUP -----------------
fig = plt.figure(figsize=(9, 8))
x = np.arange(len(nets))
width = 0.35 

ax_skew = plt.subplot2grid((2, 2), (0, 0))
ax_power = plt.subplot2grid((2, 2), (0, 1))
ax_wire = plt.subplot2grid((2, 2), (1, 0))
axes = [ax_skew, ax_power, ax_wire]

for i, ax in enumerate(axes):
    # 1. Plot MAE Bars
    b1 = ax.bar(x - width/2, mae_seen[i], width, label='MAE Seen', color='#1f77b4', edgecolor='k', alpha=0.8, zorder=3)
    b2 = ax.bar(x + width/2, mae_unseen[i], width, label='MAE Unseen', color='#ff7f0e', edgecolor='k', alpha=0.8, zorder=3)
    
    mae_ylim = (0, max(max(mae_unseen[i]), max(mae_seen[i])) * 1.5)
    ax.set_ylim(mae_ylim)
    
    # 2. Plot R2 (Secondary Axis)
    ax2 = ax.twinx()
    r2_ylim = (-3.5, 1.5)
    ax2.plot(x - width/2, r2_seen, 'D', color='red', markersize=6, label='$R^2$ Seen', markeredgecolor='k', zorder=5)
    ax2.plot(x + width/2, r2_unseen, 's', color='darkred', markersize=6, label='$R^2$ Unseen', markeredgecolor='k', zorder=5)
    ax2.set_ylim(r2_ylim)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=9)

    # 3. Smart Annotations
    for j, (rect_s, rect_u) in enumerate(zip(b1, b2)):
        smart_annotate(ax, rect_s, mae_seen[i][j], r2_seen[j], mae_ylim, r2_ylim)
        smart_annotate(ax, rect_u, mae_unseen[i][j], r2_unseen[j], mae_ylim, r2_ylim)

    ax.set_title(metrics[i], fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(nets, fontsize=10)
    ax.set_ylabel('MAE', fontsize=10, fontweight='bold')
    # Increased labelpad to move Spatial R2 away from numbers
    ax2.set_ylabel('Spatial $R^2$ (Fidelity)', fontsize=10, fontweight='bold', color='red', labelpad=20)

# Legend in bottom-right quadrant
ax_leg = plt.subplot2grid((2, 2), (1, 1))
ax_leg.axis('off')
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax_leg.legend(h1 + h2, l1 + l2, loc='center', fontsize=11, frameon=True, borderpad=1.5, labelspacing=1.2)

# Final spacing fix
plt.subplots_adjust(wspace=0.6, hspace=0.4) 

plt.savefig("clustered_mae_r2_final.pdf", format='pdf', bbox_inches='tight')
plt.savefig("clustered_mae_r2_final.png", dpi=300, bbox_inches='tight')
plt.show()