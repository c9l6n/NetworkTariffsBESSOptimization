import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import simbench as sb
import os
import time
import gc

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

def _create_results_df(results_dict):
    reinforcement_results = []

    for i in results_dict.keys():
        for j in results_dict[i].keys():
            for k in results_dict[i][j].keys():
                for l in results_dict[i][j][k].keys():
                    reinforcement_result = results_dict[i][j][k][l]["reinforcement_result"]
                    hh_attribution = results_dict[i][j][k][l]["hh_attribution"]

                    scenario_result = {
                        "grid": i,
                        **reinforcement_result
                    }
                    reinforcement_results.append(scenario_result)

    results_df = pd.DataFrame(reinforcement_results)
    results_df = results_df.sort_values(by=["grid", "seed", "tariff"])
    results_df["total_reinforcement_cost"] = (results_df["LV_total_reinforcement_cost"] 
                                          + results_df["MV_total_reinforcement_cost"] 
                                          + results_df["HV_total_reinforcement_cost"])
    
    return results_df

def _create_cumsum(results_df):
    columns_to_adjust = [
        'total_reinforcement_cost',
        'LV_total_reinforcement_cost', 'LV_total_trafo_reinforcement_cost', 'LV_total_line_reinforcement_cost',
        'MV_total_reinforcement_cost', 'MV_total_trafo_reinforcement_cost', 'MV_total_line_reinforcement_cost',
        'HV_total_reinforcement_cost', 'HV_total_trafo_reinforcement_cost', 'HV_total_line_reinforcement_cost'
    ]

    # Create incremental columns before applying cumsum
    for column in columns_to_adjust:
        inc_column = f"{column.replace('total', 'inc')}"
        results_df[inc_column] = results_df[column].copy()

    # Apply cumsum to original columns
    for column in columns_to_adjust:
        results_df[column] = results_df.groupby(['grid', 'seed', 'tariff'])[column].cumsum()

    # Replace NaN with 0 in the specified columns
    results_df[columns_to_adjust] = results_df[columns_to_adjust].fillna(0)

    # Clip negative values to 0 for the specified columns
    results_df[columns_to_adjust] = results_df[columns_to_adjust].clip(lower=0)

    return results_df

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

def _get_static_loads():
    # Build the path to the data file relative to the notebook location
    base_dir = os.path.dirname(os.path.abspath(__file__))  # for scripts
    # But since you're in a Jupyter notebook, use this instead:
    base_dir = os.path.dirname(os.getcwd())

    file_path = os.path.join(base_dir, "data", "reinforcement_files", "static_loads.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    
    else:
        grid_codes = [
            '1-LV-rural1--0-no_sw', '1-LV-rural2--0-no_sw', '1-LV-rural3--0-no_sw',
            '1-LV-semiurb4--0-no_sw', '1-LV-semiurb5--0-no_sw',
            '1-LV-urban6--0-no_sw'
        ]

        lv_results = []  # temporary storage for each grid

        for code in grid_codes:
            net = sb.get_simbench_net(code)
            static_loads = net.load[~net.load["profile"].str.startswith("H", na=False)].copy()
            total_non_hh = static_loads["p_mw"].sum()
            grid_name = code[2:-9]
            
            # append result as a dict
            lv_results.append({
                "grid": grid_name,
                "static_load": total_non_hh,
                "subgrids_static_load": 0
            })

        lv_results = pd.DataFrame(lv_results)

        grid_codes = ['1-MV-rural--0-no_sw', '1-MV-semiurb--0-no_sw',
                        '1-MV-urban--0-no_sw', '1-MV-comm--0-no_sw']
        mv_results = []

        for code in grid_codes:
            net = sb.get_simbench_net(code)
            static_loads = net.load[~net.load["profile"].str.startswith("lv_", na=False)].copy()
            total_non_hh = static_loads["p_mw"].sum()
            grid_name = code[2:-9]

            non_static = net.load[net.load["profile"].str.startswith("lv_", na=False)].copy()
            subgrids = non_static["profile"]

            lv_static_load = 0

            for sg in subgrids:
                sg = sg[:2].upper() + sg[2:].replace("_", "-")
                lv_static_load += lv_results[lv_results["grid"] == sg]["static_load"].values[0]
        
            # append result as a dict
            mv_results.append({
                "grid": grid_name,
                "static_load": total_non_hh,
                "subgrids_static_load": lv_static_load
            })

        mv_results = pd.DataFrame(mv_results)

        grid_codes = ['1-HV-mixed--0-no_sw', '1-HV-urban--0-no_sw']

        hv_results = []

        for code in grid_codes:
            net = sb.get_simbench_net(code)
            static_loads = net.load[net.load["profile"].str.startswith("mv_add", na=False)].copy()
            total_non_hh = static_loads["p_mw"].sum()
            grid_name = code[2:-9]

            non_static = net.load[~net.load["profile"].str.startswith("mv_add", na=False)].copy()
            subgrids = non_static["profile"]

            mv_static_load = 0

            for sg in subgrids:
                sg = sg[:2].upper() + sg[2:].replace("_", "-")
                mv_static_load += mv_results[mv_results["grid"] == sg]["static_load"].values[0] + mv_results[mv_results["grid"] == sg]["subgrids_static_load"].values[0]
            # append result as a dict
            hv_results.append({
                "grid": grid_name,
                "static_load": total_non_hh,
                "subgrids_static_load": mv_static_load
            })

        hv_results = pd.DataFrame(hv_results)

        # Attach all outputs
        combined_df = pd.concat([lv_results, mv_results, hv_results], ignore_index=True)
        combined_df["total_static_load"] = combined_df["static_load"] + combined_df["subgrids_static_load"]

        return combined_df

# --- Per-process cache (simple dict) to avoid reloading the same LV file repeatedly
_LV_CACHE = {}

def _load_lv(grid: str, seed: int, lv_dir_str: str):
    """Top-level, picklable loader with a small per-process in-memory cache."""
    key = (grid, seed)
    if key in _LV_CACHE:
        return _LV_CACHE[key]
    file_path = Path(lv_dir_str) / f"{grid}_seed_{seed}.pkl"
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    # keep cache bounded (evict oldest if > 64)
    if len(_LV_CACHE) >= 64:
        _LV_CACHE.pop(next(iter(_LV_CACHE)))
    _LV_CACHE[key] = obj
    return obj

# --- Your original function (unchanged); keep it TOP-LEVEL in this cell
def build_lv_block(result_lv, k, l, MAX_DISCHARGE_KW_DEFAULT = 11.5):
    # --- source HH-level data
    hh_attr = result_lv[k][l]["HH_attribution"].copy()

    # --- flags
    hh_attr["has_HP"] = hh_attr["P_HP"] > 0
    hh_attr["has_EV"] = hh_attr["P_EV"] > 0
    hh_attr["has_PV_BESS"] = (
        (hh_attr["pv_generation_kWh"] > 0) |
        (hh_attr["bess_charge_kWh"] > 0) |
        (hh_attr["bess_discharge_kWh"] > 0)
    )

    # --- households per bus (avoid /0 later)
    grouped_bus = hh_attr.groupby("bus_id").size().reset_index(name="hh_in_bus")
    hh_attr = hh_attr.merge(grouped_bus, on="bus_id", how="left")

    # --- per-HH deployed discharge capacity (vectorized)
    hh_attr["max_charging_power_deployed"] = np.where(
        hh_attr["has_PV_BESS"],
        MAX_DISCHARGE_KW_DEFAULT / hh_attr["hh_in_bus"].clip(lower=1),
        0.0
    )

    # --- sanity: required column exists
    if "import_in_cpp_events_kWh" not in hh_attr.columns:
        raise KeyError("Expected column 'import_in_cpp_events_kWh' in HH_attribution")

    # --- cut down columns
    hh_attr = hh_attr[["bus_id", "hh_id", "has_HP", "has_EV", "has_PV_BESS", "max_charging_power_deployed", 
                    "import_kWh", "window_peak_kW", "import_in_cpp_events_kWh"]]

    # --- load peak data
    peak_time = result_lv[k][l]["peak_times"]["time"].iloc[0]
    peak_times = pd.DataFrame({
        "time": result_lv[k][l]["peak_times"]["time"],
        "top_time_index": range(1, len(result_lv[k][l]["peak_times"]["time"]) + 1)
    })
    bus_peak_times = result_lv[k][l]["peak_times_df"].copy()
    hh_peak_times = result_lv[k][l]["peak_times_hh_loads"].copy()

    # --- allocate P_PV and P_BESS at peak time proportionally to deployed capacity
    hh_peak_times = hh_peak_times.merge(bus_peak_times[["bus_id", "time", "P_PV", "P_BESS", "P_HH_HP_EV"]], on=["bus_id", "time"], how="left", suffixes=("", "_bus"))
    hh_peak_times["P_PV"] = hh_peak_times["P_PV"] * (hh_peak_times["P_HH_HP_EV"]/hh_peak_times["P_HH_HP_EV_bus"])
    hh_peak_times["P_BESS"] = hh_peak_times["P_BESS"] * (hh_peak_times["P_HH_HP_EV"]/hh_peak_times["P_HH_HP_EV_bus"])
    hh_peak_times = hh_peak_times.drop(columns=["P_HH_HP_EV_bus"])
    hh_peak_times["P_Total"] = hh_peak_times["P_HH_HP_EV"] + hh_peak_times["P_PV"] + hh_peak_times["P_BESS"]

    # --- switch times with index
    hh_peak_times = hh_peak_times.merge(peak_times, on="time", how="left", suffixes=("", "_peak"))

    # --- add flags to hh_peak_times
    hh_peak_times = hh_peak_times.merge(
        hh_attr[["bus_id", "hh_id", "has_HP", "has_EV", "has_PV_BESS"]],
        on=["bus_id", "hh_id"],
        how="left"
    )
    hh_peak_times = hh_peak_times[["top_time_index", "bus_id", "hh_id", "has_HP", "has_EV", "has_PV_BESS", "P_HH_HP_EV", "P_PV", "P_BESS", "P_Total"]]

    # --- group by scenario attributes ---
    hh_attr_grouped = hh_attr.groupby(["has_HP", "has_EV", "has_PV_BESS"], as_index=False).agg(
        n_households=("hh_id", "nunique"),
        max_deployed_discharge_kW=("max_charging_power_deployed", "sum"),
        import_kWh=("import_kWh", "sum"),
        window_peak_kW=("window_peak_kW", "sum"),
        import_in_cpp_events_kWh=("import_in_cpp_events_kWh", "sum"),
    )

    hh_peak_times_grouped = hh_peak_times.groupby(["top_time_index", "has_HP", "has_EV", "has_PV_BESS"], as_index=False).agg(
        n_households=("hh_id", "nunique"),
        hh_hp_ev_kW=("P_HH_HP_EV", "sum"),
        pv_kW=("P_PV", "sum"),
        bess_kW=("P_BESS", "sum"),
        import_kW=("P_Total", "sum"),
    )
    return hh_attr_grouped, hh_peak_times_grouped

def _worker_lv_verbose(task):
    """LV-only worker for a single (i,j,k) with verbose prints."""
    i, j, k = task["i"], task["j"], task["k"]
    mv_counts_list = task["mv_counts"]
    lv_dir_str = task["lv_dir_str"]
    tariffs = ["VOL", "IPP", "CPP", "MAX"]

    t0 = time.time()
    print(f"[Worker] Start grid={i}, seed={j}, scenario={k}", flush=True)

    out_hh_attr = []
    out_peak_times = []
    try:
        for l in tariffs:
            t_l = time.time()
            lv_hh_attr = []
            lv_peak_times = []
            for grid_lv, seed_lv, n_lv in mv_counts_list:
                result_lv = _load_lv(grid_lv, seed_lv, lv_dir_str)
                hh_attr_grouped, hh_peak_times_grouped = build_lv_block(result_lv, k, l)

                scale_cols_hh_attr = [
                    "n_households", "max_deployed_discharge_kW", 
                    "import_kWh", "window_peak_kW", "import_in_cpp_events_kWh"]
                hh_attr_grouped[scale_cols_hh_attr] *= int(n_lv)

                scale_cols_peak_times = [
                    "n_households", "hh_hp_ev_kW", "pv_kW", 
                    "bess_kW", "import_kW"]
                hh_peak_times_grouped[scale_cols_peak_times] *= int(n_lv)

                lv_hh_attr.append(hh_attr_grouped)
                lv_peak_times.append(hh_peak_times_grouped)

            if not lv_hh_attr: 
                continue

            lv_hh_attr = pd.concat(lv_hh_attr, ignore_index=True)
            lv_peak_times = pd.concat(lv_peak_times, ignore_index=True)

            lv_hh_attr_grouped = lv_hh_attr.groupby(
                ["has_HP", "has_EV", "has_PV_BESS"], as_index=False).agg({
                    "n_households": "sum",
                    "max_deployed_discharge_kW": "sum",
                    "import_kWh": "sum",
                    "window_peak_kW": "sum",
                    "import_in_cpp_events_kWh": "sum",
                })
            
            lv_peak_times_grouped = lv_peak_times.groupby(
                ["top_time_index", "has_HP", "has_EV", "has_PV_BESS"], as_index=False).agg({
                    "n_households": "sum",
                    "hh_hp_ev_kW": "sum",
                    "pv_kW": "sum",
                    "bess_kW": "sum",
                    "import_kW": "sum",
                })

            try:
                HP_percentage, EV_percentage, PV_percentage = k
            except Exception:
                HP_percentage = EV_percentage = PV_percentage = k[0] if hasattr(k, "__getitem__") else k

            lv_hh_attr_grouped["grid"] = i
            lv_peak_times_grouped["grid"] = i
            lv_hh_attr_grouped["seed"] = j
            lv_peak_times_grouped["seed"] = j
            lv_hh_attr_grouped["tariff"] = l
            lv_peak_times_grouped["tariff"] = l
            lv_hh_attr_grouped["HP_percentage"] = HP_percentage
            lv_peak_times_grouped["HP_percentage"] = HP_percentage
            lv_hh_attr_grouped["EV_percentage"] = EV_percentage
            lv_peak_times_grouped["EV_percentage"] = EV_percentage
            lv_hh_attr_grouped["PV_percentage"] = PV_percentage
            lv_peak_times_grouped["PV_percentage"] = PV_percentage

            lv_hh_attr_grouped = lv_hh_attr_grouped[["grid","seed","tariff",
                                                    "HP_percentage","EV_percentage","PV_percentage",
                                                    "has_HP","has_EV","has_PV_BESS",
                                                    "n_households","max_deployed_discharge_kW",
                                                    "import_kWh", "window_peak_kW", 
                                                    "import_in_cpp_events_kWh"]]
            
            lv_peak_times_grouped = lv_peak_times_grouped[["grid","seed","tariff",
                                                          "HP_percentage","EV_percentage","PV_percentage",
                                                            "top_time_index","has_HP","has_EV", "has_PV_BESS",
                                                            "n_households","hh_hp_ev_kW","pv_kW",
                                                            "bess_kW","import_kW"]]
                                                        
            out_hh_attr.append(lv_hh_attr_grouped)
            out_peak_times.append(lv_peak_times_grouped)

            # progress for this tariff
            print(f"[Worker] grid={i}, seed={j}, scenario={k}, tariff={l} "
                  f"→ in {time.time()-t_l:.1f}s", flush=True)

            del result_lv, lv_hh_attr, lv_peak_times, lv_hh_attr_grouped, lv_peak_times_grouped
            gc.collect()

        if not out_hh_attr:
            print(f"[Worker] Done grid={i}, seed={j}, scenario={k} → empty in {time.time()-t0:.1f}s", flush=True)
            return None

        res_hh_attr = pd.concat(out_hh_attr, ignore_index=True)
        res_peak_times = pd.concat(out_peak_times, ignore_index=True)
        print(f"[Worker] Done grid={i}, seed={j}, scenario={k} "
              f"→ rows={len(res_hh_attr)} in {time.time()-t0:.1f}s", flush=True)
        return res_hh_attr, res_peak_times

    except Exception as e:
        print(f"[Worker] ERROR grid={i}, seed={j}, scenario={k}: {type(e).__name__}: {e}", flush=True)
        return None