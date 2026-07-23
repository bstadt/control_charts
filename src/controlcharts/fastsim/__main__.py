"""CLI: python -m controlcharts.fastsim run <config.yaml> [options]"""

import argparse
import logging

from .runner import run_fastsim, DEFAULT_PANEL_SIZE


def main():
    parser = argparse.ArgumentParser(prog="controlcharts.fastsim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a fastsim experiment from a config")
    run_p.add_argument("config")
    run_p.add_argument("--data-path", default=None)
    run_p.add_argument("--output-base", default="experiments/results")
    run_p.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE)
    run_p.add_argument("--skip-embed", action="store_true")
    run_p.add_argument("--skip-analysis", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.cmd == "run":
        run_dir = run_fastsim(
            config_path=args.config,
            data_path=args.data_path,
            output_base=args.output_base,
            panel_size=args.panel_size,
            skip_embed=args.skip_embed,
            skip_analysis=args.skip_analysis,
        )
        print(f"\nRun complete: {run_dir}")


if __name__ == "__main__":
    main()
