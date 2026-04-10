import sys
import argparse
from pathlib import Path
import importlib

here = Path.cwd()
project_dir = here if (here / "src").exists() else here.parent
sys.path.append(str(project_dir))

import src.generate_scenarios.generate_scenarios as generate_scenarios

import src.run_reinforcement.run_lv_reinforcement_scenario as run_lv_reinforcement_scenario
import src.run_reinforcement.run_mv_reinforcement_scenario as run_mv_reinforcement_scenario
import src.run_reinforcement.run_hv_reinforcement_scenario as run_hv_reinforcement_scenario

import src.run_reinforcement.reinforce_grid as reinforce_grid

DEFAULT_SCENARIO_SPACE = [
    [0, 0, 0],      # Base
    [4, 3, 14],     # 2024
    [30, 35, 27],   # 2030
    [42, 55, 34],   # 2035
    [52, 75, 49],   # 2040
    [72, 95, 61],   # 2045
    [100, 100, 100] # Max stress test
]

DEFAULT_SEEDS = [41, 42, 43, 44, 45]

DEFAULT_LV_CODES = [
    '1-LV-rural1--0-no_sw', '1-LV-rural2--0-no_sw', '1-LV-rural3--0-no_sw',
    '1-LV-semiurb4--0-no_sw', '1-LV-semiurb5--0-no_sw', '1-LV-urban6--0-no_sw'
]
DEFAULT_MV_CODES = [
    '1-MV-rural--0-no_sw', '1-MV-semiurb--0-no_sw',
    '1-MV-urban--0-no_sw', '1-MV-comm--0-no_sw'
]
DEFAULT_HV_CODES = [
    '1-HV-mixed--0-no_sw', '1-HV-urban--0-no_sw'
]


def parse_ints_or_ranges(s: str):
    """
    Parse strings like "41,42,45" or "41-45" or mixed "41-43,45,47-48".
    """
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_csv(s: str):
    """
    Split on commas into a list of non-empty trimmed strings.
    """
    return [x.strip() for x in s.split(",") if x.strip()]


def reload_modules():
    """
    Optional reload to mimic the notebook's importlib.reload() cells.
    """
    importlib.reload(generate_scenarios)
    importlib.reload(run_lv_reinforcement_scenario)
    importlib.reload(run_mv_reinforcement_scenario)
    importlib.reload(run_hv_reinforcement_scenario)
    importlib.reload(reinforce_grid)


def run_scenario_creation(grid_codes, seeds, scenario_space):
    """
    Run only the scenario creation pipeline (VOL/MAX/IPP/CPP etc inside your generate_scenarios.run_bess_pipeline).
    """
    for grid_code in grid_codes:
        for seed in seeds:
            print(f"[SCENARIOS] grid={grid_code} | seed={seed}")
            _ = generate_scenarios.run_bess_pipeline(grid_code, scenario_space, seed)


def run_reinforcement(level, grid_codes, seeds, scenario_space):
    """
    Run reinforcement only, for LV/MV/HV or 'all'.
    """
    def run_lv():
        for grid_code in grid_codes or DEFAULT_LV_CODES:
            for seed in seeds:
                print(f"[REINF|LV] grid={grid_code} | seed={seed}")
                run_lv_reinforcement_scenario.run_scenarios(grid_code, scenario_space, seed=seed)

    def run_mv():
        for grid_code in grid_codes or DEFAULT_MV_CODES:
            for seed in seeds:
                print(f"[REINF|MV] grid={grid_code} | seed={seed}")
                run_mv_reinforcement_scenario.run_scenarios(grid_code, scenario_space, seed=seed)

    def run_hv():
        for grid_code in grid_codes or DEFAULT_HV_CODES:
            for seed in seeds:
                print(f"[REINF|HV] grid={grid_code} | seed={seed}")
                run_hv_reinforcement_scenario.run_scenarios(grid_code, scenario_space, seed=seed)

    if level == "lv":
        run_lv()
    elif level == "mv":
        run_mv()
    elif level == "hv":
        run_hv()
    elif level == "all":
        run_lv()
        run_mv()
        run_hv()
    else:
        raise ValueError(f"Unknown reinforcement level: {level}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run BESS scenario creation and/or grid reinforcement with seeds and grid selection."
    )

    sub = p.add_subparsers(dest="mode", required=True)

    # --- scenarios subcommand ---
    ps = sub.add_parser("scenarios", help="Run scenario creation only (generate_scenarios.run_bess_pipeline)")
    ps.add_argument(
        "--grid-codes",
        type=str,
        default=",".join(DEFAULT_LV_CODES),  # default to LV set (matches the notebook first block)
        help="Comma-separated grid codes. Example: '1-LV-rural1--0-no_sw,1-LV-rural2--0-no_sw'"
    )
    ps.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds and/or ranges. Example: '41-45' or '41,42,45'"
    )
    ps.add_argument(
        "--reload",
        action="store_true",
        help="Reload modules (mimics notebook importlib.reload calls)."
    )

    # --- reinforcement subcommand ---
    pr = sub.add_parser("reinforcement", help="Run reinforcement only (LV / MV / HV)")
    pr.add_argument(
        "--level",
        choices=["lv", "mv", "hv", "all"],
        default="lv",
        help="Network level to run (default: lv)."
    )
    pr.add_argument(
        "--grid-codes",
        type=str,
        default="",  # if empty, will use defaults for chosen level
        help="Comma-separated grid codes for the chosen level. If omitted, uses built-in defaults."
    )
    pr.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds and/or ranges. Example: '41-45' or '41,42,45'"
    )
    pr.add_argument(
        "--reload",
        action="store_true",
        help="Reload modules (mimics notebook importlib.reload calls)."
    )

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "scenarios":
        seeds = parse_ints_or_ranges(args.seeds)
        grid_codes = parse_csv(args.grid_codes) if args.grid_codes else []
        if len(grid_codes) == 1 and grid_codes[0].lower() == "all":
            grid_codes = DEFAULT_LV_CODES
        if args.reload:
            reload_modules()
        print(f"Running SCENARIOS for seeds={seeds} | grids={grid_codes or '[NONE]'}")
        run_scenario_creation(grid_codes, seeds, DEFAULT_SCENARIO_SPACE)
        return

    if args.mode == "reinforcement":
        seeds = parse_ints_or_ranges(args.seeds)
        grid_codes = parse_csv(args.grid_codes) if args.grid_codes else []
        if args.reload:
            reload_modules()
        print(f"Running REINFORCEMENT level={args.level} for seeds={seeds} | grids={grid_codes or '[defaults per level]'}")
        run_reinforcement(args.level, grid_codes, seeds, DEFAULT_SCENARIO_SPACE)
        return


if __name__ == "__main__":
    main()

    # EXAMPLES OF USAGE:

    # Example usage:
    # python main.py scenarios --seeds 41-45
    # python main.py reinforcement --level all --seeds 41-45