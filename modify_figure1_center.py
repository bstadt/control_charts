#!/usr/bin/env python3
"""Modify figure1 center column to have two stacked subplots."""

with open('experiments/figures/figure1.py', 'r') as f:
    lines = f.readlines()

# Find the section to replace
output = []
in_center_section = False
skip_until_next_section = False

for i, line in enumerate(lines):
    if '# ===== (0,1): Mean accuracy comparison =====' in line:
        in_center_section = True
        skip_until_next_section = True
        
        # Insert new two-stacked-subplot code
        output.append('    # ===== (0,1): Two stacked accuracy subplots =====\n')
        output.append('    gs_acc = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.4)\n')
        output.append('\n')
        output.append('    # Compute mean accuracy across agents\n')
        output.append('    static_mean_temporal = np.mean(static_temporal_acc, axis=1)\n')
        output.append('    dynamic_mean_temporal = np.mean(dynamic_temporal_acc, axis=1)\n')
        output.append('    static_mean_nontemporal = np.mean(static_nontemporal_acc, axis=1)\n')
        output.append('    dynamic_mean_nontemporal = np.mean(dynamic_nontemporal_acc, axis=1)\n')
        output.append('\n')
        output.append('    # Top: Non-temporal accuracy\n')
        output.append('    ax_acc_nontemporal = fig.add_subplot(gs_acc[0])\n')
        output.append('    ax_acc_nontemporal.plot(static_acc_steps, static_mean_nontemporal, marker=\'o\', markersize=3,\n')
        output.append('                            linewidth=2, color=\'#EA5526\', label=\'Static\')\n')
        output.append('    ax_acc_nontemporal.plot(dynamic_acc_steps, dynamic_mean_nontemporal, marker=\'o\', markersize=3,\n')
        output.append('                            linewidth=2, color=\'#4462BD\', label=\'Dynamic\')\n')
        output.append('    ax_acc_nontemporal.set_ylabel(\'Mean Accuracy\')\n')
        output.append('    ax_acc_nontemporal.set_title(\'Accuracy on Non-temporal Questions\')\n')
        output.append('    ax_acc_nontemporal.set_ylim(-0.05, 1.05)\n')
        output.append('    ax_acc_nontemporal.legend(fontsize=7, ncol=2, loc=\'lower right\')\n')
        output.append('    ax_acc_nontemporal.grid(True, alpha=0.3)\n')
        output.append('    ax_acc_nontemporal.tick_params(labelbottom=False)  # Hide x-axis labels\n')
        output.append('\n')
        output.append('    # Bottom: Temporal accuracy\n')
        output.append('    ax_acc_temporal = fig.add_subplot(gs_acc[1])\n')
        output.append('    ax_acc_temporal.plot(static_acc_steps, static_mean_temporal, marker=\'o\', markersize=3,\n')
        output.append('                         linewidth=2, color=\'#EA5526\', label=\'Static\')\n')
        output.append('    ax_acc_temporal.plot(dynamic_acc_steps, dynamic_mean_temporal, marker=\'o\', markersize=3,\n')
        output.append('                         linewidth=2, color=\'#4462BD\', label=\'Dynamic\')\n')
        output.append('    ax_acc_temporal.set_xlabel(\'Iteration\')\n')
        output.append('    ax_acc_temporal.set_ylabel(\'Mean Accuracy\')\n')
        output.append('    ax_acc_temporal.set_title(\'Accuracy on Temporal Questions\')\n')
        output.append('    ax_acc_temporal.set_ylim(-0.05, 1.05)\n')
        output.append('    ax_acc_temporal.legend(fontsize=7, ncol=2, loc=\'lower right\')\n')
        output.append('    ax_acc_temporal.grid(True, alpha=0.3)\n')
        output.append('\n')
        continue
    
    if skip_until_next_section:
        if '# ===== (0,2):' in line:
            skip_until_next_section = False
            output.append(line)
        continue
    
    output.append(line)

with open('experiments/figures/figure1.py', 'w') as f:
    f.writelines(output)

print('✓ Modified figure1 center column to have two stacked subplots')
print('  - Top: Non-temporal accuracy (Static vs Dynamic)')
print('  - Bottom: Temporal accuracy (Static vs Dynamic)')
print('  - Using red (#EA5526) and blue (#4462BD) colors')
