#!/usr/bin/env python3
"""Regenerate figures from existing experiment results with updated styling."""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from controlcharts.tdkps_analysis import plot_combined_analysis, plot_isomirror

def regenerate_experiment_figures(results_dir: Path):
    """Regenerate figures for a single experiment result directory."""
    experiment_name = results_dir.name
    output_dir = results_dir
    snapshots_dir = results_dir / 'snapshots'
    
    # Find config file
    config_path = None
    for config_file in results_dir.glob('*.yaml'):
        config_path = config_file
        break
    
    print(f'Regenerating figures for: {experiment_name}')
    print(f'  Output dir: {output_dir}')
    print(f'  Snapshots dir: {snapshots_dir}')
    print(f'  Config: {config_path}')
    
    # Load control bar config if it exists
    control_bar_config = None
    if config_path and config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        control_bar_config = config_data.get('control_bar', None)
    
    # Regenerate combined analysis plot
    try:
        plot_combined_analysis(
            output_dir=output_dir,
            experiment_name=experiment_name,
            snapshots_dir=snapshots_dir if snapshots_dir.exists() else None,
            config_path=config_path,
            control_bar_config=control_bar_config,
        )
        print(f'  [OK] Generated combined analysis plot')
    except Exception as e:
        print(f'  [ERROR] Error generating combined analysis: {e}')
    
    # Regenerate isomirror plot
    try:
        plot_isomirror(
            output_dir=output_dir,
            experiment_name=experiment_name,
        )
        print(f'  [OK] Generated isomirror plot')
    except Exception as e:
        print(f'  [ERROR] Error generating isomirror: {e}')

def main():
    results_base = Path('experiments/results')
    
    if len(sys.argv) > 1:
        # Regenerate specific experiment
        experiment_name = sys.argv[1]
        results_dir = results_base / experiment_name
        if results_dir.exists():
            regenerate_experiment_figures(results_dir)
        else:
            print(f'Error: {results_dir} not found')
    else:
        # Regenerate all experiments
        for results_dir in sorted(results_base.iterdir()):
            if results_dir.is_dir():
                regenerate_experiment_figures(results_dir)
                print()

if __name__ == '__main__':
    main()
