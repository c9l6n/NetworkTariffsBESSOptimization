from __future__ import annotations
import os
from pathlib import Path
import pickle
from typing import Dict, Tuple

import simbench as sb
import pandas as pd

# Local modules (assumes project root is on PYTHONPATH when you launch the job)
from src.general import load_input_data
from src.generate_scenarios import generate_households
from src.generate_scenarios import generate_loads
from src.generate_scenarios import setup_scenarios
from src.generate_scenarios import calculate_bess


def run_bess_pipeline(
    grid_code: str,
    scenario_space: list[list[int]],
    seed: int,
    *,
    target_events: int = 10,
    pickle_dir: str | os.PathLike = ".",
    pickle_prefix: str | None = None,
    verbose: bool = True,
) -> Path:
    """
    End-to-end run for a single grid_code / scenario_space / seed.

    Steps:
      1) Load all input data once
      2) Load SimBench network
      3) Generate households
      4) Generate HH/HP, EV, PV loads
      5) Build setup (derives canonical scenario_list)
      6) For each (hp, ev, pv) scenario: run VOL, MAX, IPP, CPP
      7) Pickle results to disk and return the file path

    Returns
    -------
    Path
        Full path to the written pickle file.
    """
    # ---------- 1) Load input data (one time) ----------
    (
        load_data,
        household_info,
        trips,
        ev_types,
        temperature,
        roof_size,
        pv_production,
        units_per_house_probs,
        people_per_unit_probs,
        cars_per_household_probs,
        private_parking_probs,
    ) = load_input_data.load_input_data()

    # ---------- 2) Load SimBench network ----------
    net = sb.get_simbench_net(grid_code)

    # ---------- 3) Generate households ----------
    household_df = generate_households.generate_households(
        net,
        grid_code,
        household_info,
        units_per_house_probs,
        people_per_unit_probs,
        cars_per_household_probs,
        private_parking_probs,
        roof_size,
        seed,
    )

    # ---------- 4) Generate loads ----------
    hh_load_df, hp_load_df = generate_loads.generate_hh_and_hp_loads(household_df, load_data, seed)
    ev_load_df = generate_loads.generate_ev_loads(household_df, trips, temperature, ev_types, seed)
    pv_load_df = generate_loads.generate_pv_loads(household_df, pv_production, seed)

    # ---------- 5) Normalize scenarios (auto base + stress) & build setup ----------
    user_sc = scenario_space
    sc_norm = setup_scenarios._normalize_scenarios(user_sc)  # ensures (0,0,0) at start, (100,100,100) at end

    setup = setup_scenarios.build_setup(
        household_df,
        hh_load_df,
        hp_load_df,
        ev_load_df,
        pv_load_df,
        scenario_space=sc_norm,
        seed=seed,
    )

    # ---------- 6) Define time window ----------
    start = pd.Timestamp("2019-01-01 00:00:00")
    end   = pd.Timestamp("2019-12-31 23:45:00")
    time_window = (start, end)

    # ---------- 7) Run scenarios ----------
    results: Dict[Tuple[int, int, int], Dict[str, pd.DataFrame]] = {}

    for hp_p, ev_p, pv_p in setup["scenario_list"]:
        # Materialize frames for this scenario
        hh_df_scn, bus_df_scn, cap_by_bus = setup_scenarios.materialize_scenario_frames(
            setup, hp_p, ev_p, pv_p, household_df, include_pv=True
        )

        # Run algorithms (VOL, MAX, IPP, CPP)
        vol_res, bus_df_vol, soc_vol = calculate_bess.run_VOL_algorithm(bus_df_scn, hh_df_scn, cap_by_bus, time_window)
        max_res                      = calculate_bess.run_MAX_algorithm(bus_df_scn, hh_df_scn, cap_by_bus, time_window)
        ipp_res                      = calculate_bess.run_IPP_algorithm(bus_df_scn, bus_df_vol, soc_vol, hh_df_scn, cap_by_bus, time_window)
        cpp_res                      = calculate_bess.run_CPP_algorithm(bus_df_vol, soc_vol, hh_df_scn, cap_by_bus, time_window, target_events = 10)

        key = (int(hp_p), int(ev_p), int(pv_p))
        results[key] = {"VOL": vol_res, "MAX": max_res, "IPP": ipp_res, "CPP": cpp_res}

        if verbose:
            print(
                f"Finished: {grid_code} | {hp_p}% HP, {ev_p}% EV, {pv_p}% PV/BESS (seed={seed})"
            )

    # ---------- 8) Pickle results ----------
    # Reproduce your original file-naming convention
    grid_name = grid_code[2:-9]
    grid_level = grid_code[2:4]

    # --- Build folder hierarchy: project/data/scenario_files/<grid_name> ---
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "data" / "scenario_files" / grid_level
    base_dir.mkdir(parents=True, exist_ok=True)

    if pickle_prefix is None:
        pkl_name = f"{grid_name}_seed_{seed}.pkl"
    else:
        pkl_name = f"{pickle_prefix}_{grid_name}_seed_{seed}.pkl"

    pkl_path = base_dir / pkl_name

    # --- Save pickle ---
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return pkl_path
