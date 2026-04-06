#!/usr/bin/env python3
"""Update figure1.py, figure2.py, figure3.py with larger fonts and smaller heights."""

import re

files_to_update = [
    'experiments/figures/figure1.py',
    'experiments/figures/figure2.py',
    'experiments/figures/figure3.py',
]

# Backup originals
for file_path in files_to_update:
    import shutil
    shutil.copy(file_path, f'{file_path}.backup')
    print(f'Backed up {file_path}')

# Define replacements for each file
replacements = {
    'experiments/figures/figure1.py': [
        ('figsize=(15, 5)', 'figsize=(15, 4)'),
        ('fontsize=7', 'fontsize=12'),
        ('fontsize=8', 'fontsize=13'),
    ],
    'experiments/figures/figure2.py': [
        ('figsize=(14, 5)', 'figsize=(14, 4)'),
        ('fontsize=12', 'fontsize=16'),
        ('fontsize=8', 'fontsize=13'),
    ],
    'experiments/figures/figure3.py': [
        ('figsize=(14, 10)', 'figsize=(14, 8.5)'),
        ('fontsize=10', 'fontsize=16'),
        ('fontsize=11', 'fontsize=17'),
        ('fontsize=7', 'fontsize=12'),
        ('fontsize=8', 'fontsize=13'),
    ],
}

# Apply replacements
for file_path, file_replacements in replacements.items():
    with open(file_path, 'r') as f:
        content = f.read()
    
    for old, new in file_replacements:
        content = content.replace(old, new)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f'Updated {file_path}')

print('\nAll figure scripts updated with:')
print('  - Larger font sizes')
print('  - Smaller figure heights')
