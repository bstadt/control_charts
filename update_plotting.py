#!/usr/bin/env python3
"""Update tdkps_analysis.py with larger fonts and smaller heights."""

import re

file_path = 'src/controlcharts/tdkps_analysis.py'

with open(file_path, 'r') as f:
    content = f.read()

# Update figure sizes (reduce y dimension)
replacements = [
    # run_tdkps_analysis
    ('figsize=(10, 6)', 'figsize=(10, 5)'),
    # plot_perspective_variance  
    # (already has figsize=(10, 6), covered above)
    # plot_combined_analysis - 3x3
    ('figsize=(18, 14)', 'figsize=(18, 12)'),
    # plot_combined_analysis - 2x2
    ('figsize=(14, 10)', 'figsize=(14, 8.5)'),
    # plot_isomirror
    ('figsize=(18, 5)', 'figsize=(18, 4.5)'),
]

# Update font sizes (increase all)
font_replacements = [
    ('fontsize=6', 'fontsize=11'),
    ('fontsize=7', 'fontsize=12'),
    ('fontsize=8', 'fontsize=13'),
    ('fontsize=10', 'fontsize=15'),
    ('fontsize=14', 'fontsize=20'),
]

# Apply replacements
for old, new in replacements:
    content = content.replace(old, new)

for old, new in font_replacements:
    content = content.replace(old, new)

# Also increase default matplotlib font sizes by adding rcParams at top of plotting functions
# Add after the first plt.figure() call in each function
imports_section = '''import numpy as np
import matplotlib.pyplot as plt'''

updated_imports = '''import numpy as np
import matplotlib.pyplot as plt

# Set larger default font sizes for all plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 20
})'''

content = content.replace(imports_section, updated_imports)

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print('✓ Updated tdkps_analysis.py with:')
print('  - Larger font sizes (6→11, 7→12, 8→13, 10→15, 14→20)')
print('  - Smaller figure heights (reduced y dimensions by 10-20%)')
print('  - Increased default matplotlib rcParams for all text')
