import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import simbench as sb

def _load_all_results(voltage_level="HV"):
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "data" / "reinforcement_files" / voltage_level
    results_dict = {}

    for file_path in base_dir.glob("*.pkl"):
        fname = file_path.stem  # e.g. "LV_urban6_seed_41"
        try:
            grid_type, _, seed_str = fname.rsplit("_", 2)
            seed = int(seed_str)
        except ValueError:
            print(f"[Skipping unexpected file name: {fname}")
            continue

        # Load pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Organize into dict
        results_dict.setdefault(grid_type, {})[seed] = data

    return results_dict

def _make_label(name, hp, ev, pv, pv_label="PV/BESS"):
    if name:
        return f"{name} ({hp}% HP, {ev}% EV, {pv}% {pv_label})"
    return f"{hp}% HP, {ev}% EV, {pv}% {pv_label}"

def _assign_scenarios(
    df,
    scenarios=None,
    cols=("HP_percentage", "EV_percentage", "PV_percentage"),
    pv_label="PV/BESS",
    tolerance=None,  # e.g. 1 or 2 -> rounds to nearest tolerance for matching
):
    """
    If `scenarios` is provided (list of dicts with keys: name, hp, ev, pv):
      - Left-merge labels onto df based on (HP, EV, PV), optionally with rounding.
    Else:
      - Just build labels from the row's percentages.
    """

    hp_col, ev_col, pv_col = cols

    if scenarios is None:
        # Fallback: format direct from percentages
        df = df.copy()
        df["scenario"] = (
            df[cols]
            .astype(int)
            .apply(lambda r: _make_label(None, r[hp_col], r[ev_col], r[pv_col], pv_label), axis=1)
        )
        return df

    # Build scenario table
    scen_df = pd.DataFrame(scenarios).copy()
    # Normalize column names
    rename_map = {
        "hp": hp_col, "ev": ev_col, "pv": pv_col,
        "HP": hp_col, "EV": ev_col, "PV": pv_col
    }
    scen_df = scen_df.rename(columns=rename_map)

    # Ensure required columns exist
    for c in [hp_col, ev_col, pv_col]:
        if c not in scen_df.columns:
            raise ValueError(f"scenario table missing column: {c}")

    # Fill optional name column
    if "name" not in scen_df.columns:
        scen_df["name"] = None

    # Make label
    scen_df["scenario"] = scen_df.apply(
        lambda r: _make_label(r["name"], int(r[hp_col]), int(r[ev_col]), int(r[pv_col]), pv_label),
        axis=1
    )

    # Prepare copies for merge
    left = df.copy()
    right = scen_df[[hp_col, ev_col, pv_col, "scenario"]].copy()

    if tolerance is not None and tolerance > 0:
        # Round both sides to nearest `tolerance` for matching
        def round_to(x, tol):  # works for int/float
            return (np.round(np.asarray(x, dtype=float) / tol) * tol).astype(int)

        for c in [hp_col, ev_col, pv_col]:
            left[c + "__key"] = round_to(left[c], tolerance)
            right[c + "__key"] = round_to(right[c], tolerance)

        merge_cols = [c + "__key" for c in [hp_col, ev_col, pv_col]]
    else:
        # Exact match on the raw columns
        merge_cols = [hp_col, ev_col, pv_col]

    # Do the merge
    merged = left.merge(
        right.rename(columns={merge_cols[0]: merge_cols[0], merge_cols[1]: merge_cols[1], merge_cols[2]: merge_cols[2]}),
        how="left",
        left_on=merge_cols,
        right_on=merge_cols,
        suffixes=("", "_scen")
    )

    # Fallback label for unmatched rows
    if "scenario" not in merged.columns:
        merged["scenario"] = np.nan

    mask_unmatched = merged["scenario"].isna()
    if mask_unmatched.any():
        merged.loc[mask_unmatched, "scenario"] = (
            merged.loc[mask_unmatched, [hp_col, ev_col, pv_col]]
            .astype(int)
            .apply(lambda r: _make_label(None, r[hp_col], r[ev_col], r[pv_col], pv_label), axis=1)
        )

    # Clean temporary keys
    if tolerance is not None and tolerance > 0:
        merged.drop(columns=[c + "__key" for c in [hp_col, ev_col, pv_col]], inplace=True)

    return merged

def get_non_hh_grid_loads():
    grid_codes = [
        '1-LV-rural1--0-no_sw', '1-LV-rural2--0-no_sw', '1-LV-rural3--0-no_sw',
        '1-LV-semiurb4--0-no_sw', '1-LV-semiurb5--0-no_sw',
        '1-LV-urban6--0-no_sw'
    ]

    results = []  # temporary storage for each grid

    for code in grid_codes:
        net = sb.get_simbench_net(code)
        static_loads = net.load[~net.load["profile"].str.startswith("H", na=False)].copy()
        total_non_hh = static_loads["p_mw"].sum()
        grid_name = code[2:-9]
        
        # append result as a dict
        results.append({
            "grid": grid_name,
            "non_hh_load": total_non_hh
        })

    grid_codes = ['1-MV-rural--0-no_sw', '1-MV-semiurb--0-no_sw',
                '1-MV-urban--0-no_sw', '1-MV-comm--0-no_sw']

    for code in grid_codes:
        net = sb.get_simbench_net(code)
        static_loads = net.load[~net.load["profile"].str.startswith("lv_", na=False)].copy()
        total_non_hh = static_loads["p_mw"].sum()
        grid_name = code[2:-9]
        
        # append result as a dict
        results.append({
            "grid": grid_name,
            "non_hh_load": total_non_hh
        })

    grid_codes = ['1-HV-mixed--0-no_sw', '1-HV-urban--0-no_sw']

    for code in grid_codes:
        net = sb.get_simbench_net(code)
        static_loads = net.load[net.load["profile"].str.startswith("mv_add", na=False)].copy()
        total_non_hh = static_loads["p_mw"].sum()
        grid_name = code[2:-9]
        
        # append result as a dict
        results.append({
            "grid": grid_name,
            "non_hh_load": total_non_hh
        })

    # convert to DataFrame
    grid_loads_df = pd.DataFrame(results)

    return grid_loads_df