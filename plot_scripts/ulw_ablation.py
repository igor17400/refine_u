import matplotlib.pyplot as plt
import numpy as np

ulw_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

# Non-gated: MAE, MSE
non_gated_mae = [0.13784, 0.13797, 0.13802, 0.13803, 0.13810, 0.13812]
non_gated_mse = [0.03066, 0.03072, 0.03075, 0.03075, 0.03077, 0.03078]

# Gated: MAE, MSE
gated_mae = [0.13823, 0.13781, 0.13786, 0.13791, 0.13794, 0.13803]
gated_mse = [0.03086, 0.03064, 0.03068, 0.03069, 0.03071, 0.03075]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# MAE plot
ax1.plot(ulw_values, non_gated_mae, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Non-gated')
ax1.plot(ulw_values, gated_mae, 's--', color='#F44336', linewidth=2, markersize=8, label='Gated')
ax1.set_xlabel('UNet Loss Weight ($\\lambda$)', fontsize=12)
ax1.set_ylabel('MAE $\\downarrow$', fontsize=12)
ax1.set_title('MAE vs UNet Loss Weight', fontsize=13, fontweight='bold')
ax1.set_xticks(ulw_values)
ax1.set_xticklabels(['0.0', '0.05', '0.1', '0.2', '0.5', '1.0'])
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(axis='y', style='plain', useOffset=False)

# MSE plot
ax2.plot(ulw_values, non_gated_mse, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Non-gated')
ax2.plot(ulw_values, gated_mse, 's--', color='#F44336', linewidth=2, markersize=8, label='Gated')
ax2.set_xlabel('UNet Loss Weight ($\\lambda$)', fontsize=12)
ax2.set_ylabel('MSE $\\downarrow$', fontsize=12)
ax2.set_title('MSE vs UNet Loss Weight', fontsize=13, fontweight='bold')
ax2.set_xticks(ulw_values)
ax2.set_xticklabels(['0.0', '0.05', '0.1', '0.2', '0.5', '1.0'])
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(axis='y', style='plain', useOffset=False)

plt.tight_layout()
plt.savefig('plots/ulw_ablation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('plots/ulw_ablation.png', bbox_inches='tight', dpi=300)
print('Saved to plots/ulw_ablation.pdf and plots/ulw_ablation.png')
