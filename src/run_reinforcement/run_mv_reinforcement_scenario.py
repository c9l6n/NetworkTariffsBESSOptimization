import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Iterable, Union, Literal
import random
import simbench as sb

from src.run_reinforcement import reinforce_grid


def _load_all_results(voltage_level="LV"):
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

def _extract_field_df(
    results_dict: dict,
    scenario: Tuple[int, int, int],
    tariffs: Optional[Union[str, Iterable[str]]] = None,
    fields: Union[str, Iterable[str]] = "scenario_load",
    grids: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    include_missing: bool = False,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of one or more reinforcement_result fields
    for a specific (HP, EV, PV) across tariffs, grids, and seeds.

    Parameters
    ----------
    results_dict : dict
        Nested mapping like results_dict[grid][seed] -> results,
        where results[(hp,ev,pv)][tariff]["reinforcement_result"][field] exists.
    scenario : (int,int,int)
        (HP%, EV%, PV%), e.g. (14, 25, 37)
    tariffs : None | str | Iterable[str]
        - None: include *all* tariffs found under that scenario
        - str: include just this tariff (e.g. "VOL")
        - Iterable[str]: include only these (e.g. ["VOL","MAX","IPP","CPP"])
    fields : str | Iterable[str]
        One or more keys to extract from reinforcement_result.
    grids, seeds : optional
        Restrict to these grids or seeds.
    include_missing : bool
        If True, insert rows with NaN for missing tariffs/fields; else skip.

    Returns
    -------
    pd.DataFrame with columns: ["grid", "seed", "tariff", *fields]
    """
    hp, ev, pv = map(int, scenario)
    scen_key = (hp, ev, pv)

    grid_filter = set(grids) if grids is not None else None
    seed_filter = set(seeds) if seeds is not None else None

    # Normalize fields input
    if isinstance(fields, str):
        field_list = [fields]
    else:
        field_list = list(fields)

    rows = []
    for grid, seeds_map in results_dict.items():
        if grid_filter is not None and grid not in grid_filter:
            continue

        for seed, res in seeds_map.items():
            if seed_filter is not None and seed not in seed_filter:
                continue

            if scen_key not in res:
                if include_missing:
                    t_iter = (
                        [tariffs] if isinstance(tariffs, str)
                        else list(tariffs) if tariffs is not None
                        else [np.nan]
                    )
                    for t in t_iter:
                        row = {"grid": grid, "seed": seed, "tariff": t}
                        for f in field_list:
                            row[f] = np.nan
                        rows.append(row)
                continue

            scen_map = res[scen_key]

            # decide which tariffs to extract
            if tariffs is None:
                t_iter = list(scen_map.keys())
            elif isinstance(tariffs, str):
                t_iter = [tariffs]
            else:
                t_iter = list(tariffs)

            for t in t_iter:
                try:
                    rr = scen_map[t]["reinforcement_result"]
                except KeyError:
                    if include_missing:
                        rr = {}
                    else:
                        continue

                row = {"grid": grid, "seed": seed, "tariff": t}
                for f in field_list:
                    row[f] = rr.get(f, np.nan)
                rows.append(row)

    return pd.DataFrame(rows).sort_values(["grid", "seed", "tariff"]).reset_index(drop=True)

def _build_load_template(net):

    net.load["profile"] = net.load["profile"].fillna("").astype(str)
    static_loads = net.load[~net.load["profile"].str.startswith("lv_", na=False)].copy()  # to remain untouched
    update_candidates = net.load[net.load["profile"].str.startswith("lv_", na=False)].copy()  # lv grid loads

    template_base = (
        update_candidates
        .drop(columns=["p_mw", "q_mvar", "sn_mva"], errors="ignore")
        .drop_duplicates(subset="bus", keep="first")
        .reset_index(drop=True)
    )

    template_base["bus"] = template_base["bus"].astype(int)

    return template_base, static_loads

def _make_lv_summary_dict(df: pd.DataFrame) -> dict:
    """
    Summarize one (grid, seed, tariff) slice by summing numeric fields.
    Assumes df is already filtered to that slice.
    """
    if df.empty:
        raise ValueError("df is empty; filter to one (grid, seed, tariff) slice first.")

    # carry IDs from the first row
    out = {
        "grid_code":     df["grid"].iloc[0],
        "seed":          int(df["seed"].iloc[0])
    }

    # sum everything you listed (since you said no averages)
    out.update({
        "total_households":                     int(df["total_households"].sum()),
        "LV_total_reinforcement_cost":          float(df["total_reinforcement_cost"].sum()),
        "LV_total_trafo_reinforcement_cost":    float(df["total_trafo_reinforcement_cost"].sum()),
        "LV_total_line_reinforcement_cost":     float(df["total_line_reinforcement_cost"].sum()),
        "LV_base_#_of_transformers":            int(df["base_trafo_count"].sum()),
        "LV_scen_#_of_crit_transformers":         int(df["scen_#_of_crit_transformers"].sum()),
        "LV_scen_#_of_transformers":              int(df["scen_#_of_transformers"].sum()),
        "LV_base_km_of_lines":                  float(df["base_km_of_lines"].sum()),
        "LV_scen_km_of_crit_lines":               float(df["scen_km_of_crit_lines"].sum()),
        "LV_scen_new_km_of_lines":                float(df["scen_new_km_of_lines"].sum()),
    })
    return out

def _mv_assign_load_to_network(results_dict, load_template, static_loads, load_input, scen_idx
                               ) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    tan_phi = np.tan(np.arccos(0.95))
    template = load_template.copy()
    dict_rows = []
    bus_rows = []

    static_loads_subgrids = 0.0  # MW

    for idx, row in template.iterrows():
        level, grid_type = row["profile"].split("_", 1)
        grid = f"{level.upper()}-{grid_type}"

        sub = load_input.loc[load_input["grid"] == grid]
        if sub.empty:
            raise KeyError(f"No rows match grid={grid!r}")

        val = sub.sample(n=1).iloc[0] if len(sub) > 1 else sub.iloc[0]

        # pull reinforcement_result dict
        rr = results_dict[val["grid"]][val["seed"]][scen_idx][val["tariff"]]['reinforcement_result']
        rr = {"mv_bus": row["bus"], "grid": grid, **rr}
        dict_rows.append(rr)

        static_loads_subgrids += rr["non_hh_load"] / 1000.0  # MW

        # set load values on template
        p_mw = float(rr["scenario_load"]) / 1000.0
        q_mvar = p_mw * tan_phi
        sn_mva = np.hypot(p_mw, q_mvar)
        template.loc[idx, ["p_mw", "q_mvar", "sn_mva"]] = [p_mw, q_mvar, sn_mva]

        bus_rows.append({
            "mv_bus": int(row["bus"]),
            "grid":   val["grid"],
            "seed":   int(val["seed"])
        })

    netload_new = pd.concat([static_loads, template], ignore_index=True)
    df_dict = pd.DataFrame(dict_rows)
    lv_reinforcement_dict = _make_lv_summary_dict(df_dict)

    hh_index = pd.DataFrame(bus_rows)

    return netload_new, lv_reinforcement_dict, hh_index, static_loads_subgrids

def _save_results(results, grid_code, seed):
    # Extract grid name in same way as inside run_scenarios
    grid_name = grid_code[2:-9]
    grid_level = grid_code[2:4]

    # Build folder path
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "data" / "reinforcement_files" / grid_level
    base_dir.mkdir(parents=True, exist_ok=True)

    # File name
    pkl_name = f"{grid_name}_seed_{seed}.pkl"
    file_path = base_dir / pkl_name

    # Save pickle
    with open(file_path, "wb") as f:
        pickle.dump(results, f)

# ----------------------------
# Main MV reinforcement loop
# ----------------------------

def _reinforce_mv_grid(
        net, grid_code, lv_results_dict, load_input, load_template, static_loads, scen_idx, seed
        ):

    reinforce_grid._set_random_seed(seed)

    trafo_cost = 0
    line_cost = 0
    max_trafo_load = 0
    max_crit_trafos = 0
    max_crit_lines_km = 0
    topMW = 0

    # Replace loads 
    netload_new, lv_reinforcement_dict, hh_data, static_loads_subgrids = _mv_assign_load_to_network(lv_results_dict, load_template, static_loads, load_input, scen_idx)

    total_load = netload_new['p_mw'].sum()
    topMW = total_load

    # Run reinforcement logic
    (t_cost, l_cost, trafo_load, crit_trafos,
        crit_lines_km) = reinforce_grid._run_reinforcement(net, grid_code, netload_new, "MV")

    # Track max values and accumulate cost
    trafo_cost += t_cost
    line_cost += l_cost
    max_trafo_load = max(max_trafo_load, trafo_load)
    max_crit_trafos = max(max_crit_trafos, crit_trafos)
    max_crit_lines_km = max(max_crit_lines_km, crit_lines_km)

    return (trafo_cost, line_cost, topMW,
            max_trafo_load, max_crit_trafos,
            max_crit_lines_km, lv_reinforcement_dict, hh_data, static_loads_subgrids)

# ----------------------------
# Run scenarios
# ----------------------------

def run_scenarios(
    grid_code,
    scenario_space,
    tariff_space = [
        "VOL",        # Volumetric tariff
        "MAX",        # Maximum Potential
        "IPP",        # Individual Peak Pricing tariff
        "CPP"         # Critical Peak Pricing tariff
    ],
    seed = 42
    ) -> dict:

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    lv_results_dict = _load_all_results(voltage_level="LV")

    results = {}

    # Reload grid to start new
    net = sb.get_simbench_net(grid_code)

    # build load templates
    load_template, static_loads = _build_load_template(net)

    for tariff in tariff_space:

        # Load grid only once per reinforcement level
        net = sb.get_simbench_net(grid_code)
        net.sgen.drop(net.sgen.index, inplace=True)
        net.storage.drop(net.storage.index, inplace=True)

        base_trafo_count = len(net.trafo)
        base_line_km = net.line["length_km"].sum()
        base_load = 1e15  # start high to find minimum

        for i, (hp, ev, pv) in enumerate(scenario_space):
            
            scen_idx = (hp, ev, pv)

            # import load data
            load_input = _extract_field_df(lv_results_dict, scenario=(hp, ev, pv), tariffs=tariff, fields=["scenario_load", "non_hh_load"])

            (trafo_cost, line_cost, topMW,
              max_trafo_load, max_crit_trafo, max_crit_lines_km, 
              lv_reinforcement_dict, hh_data, static_loads_subgrids) = _reinforce_mv_grid(net, grid_code, lv_results_dict, load_input, load_template, static_loads, scen_idx, seed)
            
            total_reinforcement_cost = trafo_cost + line_cost

            if i == 0:
                base_load = min(base_load, topMW*1000)

            non_hh_load = static_loads["p_mw"].sum()*1000 + static_loads_subgrids*1000

            reinforcement_result = {
                'seed': seed,
                'tariff': tariff,
                'HP_percentage': hp,
                'EV_percentage': ev,
                'PV_percentage': pv,
                'total_households': lv_reinforcement_dict["total_households"],
                'LV_total_reinforcement_cost': lv_reinforcement_dict["LV_total_reinforcement_cost"],
                'LV_total_trafo_reinforcement_cost': lv_reinforcement_dict["LV_total_trafo_reinforcement_cost"],
                'LV_total_line_reinforcement_cost': lv_reinforcement_dict["LV_total_line_reinforcement_cost"],
                'MV_total_reinforcement_cost': total_reinforcement_cost,
                'MV_total_trafo_reinforcement_cost': trafo_cost,
                'MV_total_line_reinforcement_cost': line_cost,
                'base_load': base_load,
                'scenario_load': topMW*1000,
                'hh_load': topMW*1000 - non_hh_load,
                'non_hh_load': non_hh_load,
                'max_trafo_load': max_trafo_load,
                'LV_base_#_of_transformers': lv_reinforcement_dict["LV_base_#_of_transformers"],
                'LV_scen_#_of_crit_transformers': lv_reinforcement_dict["LV_scen_#_of_crit_transformers"],
                'LV_scen_#_of_transformers': lv_reinforcement_dict["LV_scen_#_of_transformers"],
                'LV_base_km_of_lines': lv_reinforcement_dict["LV_base_km_of_lines"],
                'LV_scen_km_of_crit_lines': lv_reinforcement_dict["LV_scen_km_of_crit_lines"],
                'LV_scen_new_km_of_lines': lv_reinforcement_dict["LV_scen_new_km_of_lines"],
                'MV_base_#_of_transformers': base_trafo_count,
                'MV_scen_#_of_crit_transformers': max_crit_trafo,
                'MV_scen_#_of_transformers': len(net.trafo),
                'MV_base_km_of_lines': base_line_km,
                'MV_scen_km_of_crit_lines': max_crit_lines_km,
                'MV_scen_new_km_of_lines': net.line["length_km"].sum()
            }

            scenario_results = {
                "reinforcement_result": reinforcement_result,  # singular name
                "hh_attribution": hh_data
            }

            # Iteratively add to nested results
            results.setdefault(scen_idx, {})[tariff] = scenario_results

    _save_results(results, grid_code, seed)