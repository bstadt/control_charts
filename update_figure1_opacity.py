#!/usr/bin/env python3
"""Make figure1 lines opaque and reduce height."""

with open('experiments/figures/figure1.py', 'r') as f:
    content = f.read()

# Remove alpha=0.7 from non-temporal lines
content = content.replace(
    "linewidth=2, linestyle='--', color='#51915B', alpha=0.7, label='Static (Non-temporal)'",
    "linewidth=2, linestyle='--', color='#51915B', label='Static (Non-temporal)'"
)
content = content.replace(
    "linewidth=2, linestyle='--', color='#8064A2', alpha=0.7, label='Dynamic (Non-temporal)'",
    "linewidth=2, linestyle='--', color='#8064A2', label='Dynamic (Non-temporal)'"
)

# Reduce figure height
content = content.replace('figsize=(15, 5)', 'figsize=(15, 3.5)')

with open('experiments/figures/figure1.py', 'w') as f:
    f.write(content)

print('✓ Made all lines opaque')
print('✓ Reduced figure height to 3.5')
