#!/usr/bin/env python3
"""Update figure1 colors and figure3 legend backgrounds."""

# Update figure1.py - change center panel to use 4 different colors
with open('experiments/figures/figure1.py', 'r') as f:
    content = f.read()

# Replace the non-temporal colors to use green and purple
content = content.replace(
    "linewidth=2, linestyle='--', color='#EA5526', alpha=0.7, label='Static (Non-temporal)'",
    "linewidth=2, linestyle='--', color='#51915B', alpha=0.7, label='Static (Non-temporal)'"
)
content = content.replace(
    "linewidth=2, linestyle='--', color='#4462BD', alpha=0.7, label='Dynamic (Non-temporal)'",
    "linewidth=2, linestyle='--', color='#8064A2', alpha=0.7, label='Dynamic (Non-temporal)'"
)

with open('experiments/figures/figure1.py', 'w') as f:
    f.write(content)

print('✓ Updated figure1.py with 4 colors from calco-5 palette')

# Update figure3.py - make legend backgrounds opaque
with open('experiments/figures/figure3.py', 'r') as f:
    content = f.read()

# Add framealpha=1 to all legend calls
import re
content = re.sub(
    r'(ax\.legend\([^)]+)\)',
    r'\1, framealpha=1)',
    content
)

with open('experiments/figures/figure3.py', 'w') as f:
    f.write(content)

print('✓ Updated figure3.py with opaque legend backgrounds')
