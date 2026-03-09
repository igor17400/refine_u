import matplotlib.pyplot as plt
import numpy as np

ulw_values = [0.0, 0.05, 0.1, 0.2]

# RefineU Betweenness (non-gated, LReLU) — UNet loss hurts
hurt_mae = [0.13784, 0.13797, 0.13802, 0.13803]

# RefineU Betweenness Gated (LReLU) — UNet loss helps
help_mae = [0.13823, 0.13781, 0.13786, 0.13791]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(ulw_values, hurt_mae, 'o-', color='#2196F3', linewidth=2, markersize=8, label='RefineU-B')
ax.plot(ulw_values, help_mae, 's--', color='#F44336', linewidth=2, markersize=8, label='RefineU-B-Gated')
ax.set_xlabel('UNet Loss Weight ($\\lambda$)', fontsize=12)
ax.set_ylabel('MAE $\\downarrow$', fontsize=12)
ax.set_title('MAE vs UNet Loss Weight', fontsize=13, fontweight='bold')
ax.set_xticks(ulw_values)
ax.set_xticklabels(['0.0', '0.05', '0.1', '0.2'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.ticklabel_format(axis='y', style='plain', useOffset=False)

plt.tight_layout()
plt.savefig('plots/ulw_ablation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('plots/ulw_ablation.png', bbox_inches='tight', dpi=300)
print('Saved to plots/ulw_ablation.pdf and plots/ulw_ablation.png')
