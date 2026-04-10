import pandas as pd
import numpy as np
import simbench as sb
import random
import os
from pathlib import Path
import pickle

from src.general import load_input_data
from src.generate_scenarios import generate_households
from src.run_reinforcement import reinforce_grid

def _create_scenario_df(scenarios, scenario_space, tariff):
    scenario_list = []
    keys_percentages = scenario_space

    # Process each scenario
    for (hp_pct, ev_pct, pv_pct) in keys_percentages:
        df = scenarios[(hp_pct, ev_pct, pv_pct)][tariff]["peak_times_df"]
        peak = df.groupby('time').agg({'P_Total': 'sum'})['P_Total'].max()

        scenario_list.append({
            'hp_percentage': hp_pct,
            'ev_percentage': ev_pct,
            'pv_percentage': pv_pct,
            'peak': peak
        })

    scenario_df = pd.DataFrame(scenario_list).reset_index(drop=True)

    return scenario_df

def _build_load_template(net, household_lookup):

    net.load["profile"] = net.load["profile"].fillna("").astype(str)
    static_loads = net.load[~net.load["profile"].str.startswith("H", na=False)].copy()  # to remain untouched
    update_candidates = net.load[net.load["profile"].str.startswith("H", na=False)].copy()  # H-loads

    template_base = (
        update_candidates
        .drop(columns=["p_mw", "q_mvar", "sn_mva"], errors="ignore")
        .drop_duplicates(subset="bus", keep="first")
        .reset_index(drop=True)
    )

    template_base["bus"] = template_base["bus"].astype(int)

    # Precompute profile and name
    bus_ids = template_base["bus"].to_numpy()
    lookup_vals = [household_lookup.get(b, {"num_units": 0, "num_people": 0}) for b in bus_ids]
    num_units = np.fromiter((v["num_units"] for v in lookup_vals), dtype=int)
    num_people = np.fromiter((v["num_people"] for v in lookup_vals), dtype=int)

    template_base["profile"] = np.char.add(num_units.astype(str), "_") + num_people.astype(str)
    subnet = template_base["subnet"].iloc[0] if "subnet" in template_base.columns else "Unknown"
    template_base["name"] = np.char.add(f"{subnet} Load ", bus_ids.astype(str))

    return template_base, static_loads

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

def _lv_assign_load_to_network(profiles_time, load_template, static_loads):
    
    # Vectorized load calculations
    p_mw = profiles_time["P_Total"].to_numpy() / 1000
    q_mvar = p_mw * np.tan(np.arccos(0.95))
    sn_mva = np.sqrt(p_mw**2 + q_mvar**2)

    # Minimal dataframe copy with set index
    df = profiles_time[["bus_id"]].copy()
    df["p_mw"] = p_mw
    df["q_mvar"] = q_mvar
    df["sn_mva"] = sn_mva
    df.set_index("bus_id", inplace=True)

    # Reindex template 
    template_indexed = load_template.set_index("bus")
    load_updates = template_indexed.copy()
    load_updates[["p_mw", "q_mvar", "sn_mva"]] = df.to_numpy()

    # Recombine with static loads
    load_updates.reset_index(inplace=True)
    netload_new = pd.concat([static_loads, load_updates], ignore_index=True)

    return netload_new

# ----------------------------
# Main reinforcement loop
# ----------------------------

def _reinforce_lv_grid(
        net, grid_code, top_times_rank, profiles_by_time, load_template, static_loads, seed
        ):

    reinforce_grid._set_random_seed(seed)

    trafo_cost = 0
    line_cost = 0
    max_trafo_load = 0
    max_crit_trafos = 0
    max_crit_lines_km = 0
    topMW = 0

    # Loop through peak times
    for rank, specific_time in enumerate(top_times_rank, start=1):
        profiles_time = profiles_by_time[specific_time]

        # Call the optimized assign_load_to_network
        netload_new = _lv_assign_load_to_network(profiles_time, load_template, static_loads)
        total_load = netload_new['p_mw'].sum()

        if rank == 1:  # Capture the highest load
            topMW = total_load

        # Run reinforcement logic
        (t_cost, l_cost, trafo_load, crit_trafos,
         crit_lines_km) = reinforce_grid._run_reinforcement(net, grid_code, netload_new, grid_level = "LV")

        # Track max values and accumulate cost
        trafo_cost += t_cost
        line_cost += l_cost
        max_trafo_load = max(max_trafo_load, trafo_load)
        max_crit_trafos = max(max_crit_trafos, crit_trafos)
        max_crit_lines_km = max(max_crit_lines_km, crit_lines_km)

    return (trafo_cost, line_cost, topMW,
            max_trafo_load, max_crit_trafos,
            max_crit_lines_km)

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

    grid_name = grid_code[2:-9]
    grid_level = grid_code[2:4]
    pkl_name = f"{grid_name}_seed_{seed}"

    # Build folder hierarchy: project/data/scenario_files/grid_level
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "data" / "scenario_files" / grid_level
    base_dir.mkdir(parents=True, exist_ok=True)
    pkl_name = f"{grid_name}_seed_{seed}.pkl"

    file_path = base_dir / pkl_name

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as f:
            scenarios = pickle.load(f)
    else:
        print("File not found or empty:", file_path)
        
    results = {}

    # Reload grid to start new
    net = sb.get_simbench_net(grid_code)

    # Load input data once - this must be outside of the generation of loads per scenario
    (load_data, household_info, 
    trips, ev_types, temperature,
    roof_size, pv_production,
    units_per_house_probs, people_per_unit_probs, cars_per_household_probs, private_parking_probs) = load_input_data.load_input_data()

    household_df = generate_households.generate_households(net, grid_code, household_info, units_per_house_probs, people_per_unit_probs, cars_per_household_probs, private_parking_probs, roof_size, seed)

    # Precompute household_lookup once
    household_lookup = (
        household_df.groupby("bus_id")
        .agg(num_units=('bus_id', 'count'), num_people=('num_people', 'sum'))
        .to_dict("index")
    )

    # build load templates
    load_template, static_loads = _build_load_template(net, household_lookup)

    for tariff in tariff_space:
        scenario_df = _create_scenario_df(scenarios, scenario_space, tariff)

        # Load grid only once per reinforcement level
        net = sb.get_simbench_net(grid_code)
        net.sgen.drop(net.sgen.index, inplace=True)
        net.storage.drop(net.storage.index, inplace=True)

        base_load = scenario_df.iloc[0]["peak"]
        base_trafo_count = len(net.trafo)
        base_line_km = net.line["length_km"].sum()

        base_load = 1e15  # start high to find minimum

        for i, row in enumerate(scenario_df.itertuples()):
            hp = int(row.hp_percentage)
            ev = int(row.ev_percentage)
            pv = int(row.pv_percentage)
            scen_idx = (hp, ev, pv)

            # Scenario inputs
            scenario = scenarios[scen_idx][tariff]  # casted scen_idx to match key types
            profiles_by_time_peaks = {
                t: df for t, df in scenario["peak_times_df"].groupby("time")
            }
            top_times_rank = scenario["peak_times"]["time"].tolist()

            (trafo_cost, line_cost, topMW,
              max_trafo_load, max_crit_trafo, max_crit_lines_km) = _reinforce_lv_grid(net, grid_code, top_times_rank, profiles_by_time_peaks, load_template, static_loads, seed)

            if i == 0:
                base_load = min(base_load, topMW*1000)

            total_reinforcement_cost = trafo_cost + line_cost
            non_hh_load = static_loads["p_mw"].sum()*1000
            
            reinforcement_result = {
                'seed': seed,
                'tariff': tariff,
                'HP_percentage': hp,
                'EV_percentage': ev,
                'PV_percentage': pv,
                'total_households': household_df["hh_id"].nunique(),
                'total_reinforcement_cost': total_reinforcement_cost,
                'total_trafo_reinforcement_cost': trafo_cost,
                'total_line_reinforcement_cost': line_cost,
                'base_load': base_load,
                'scenario_load': topMW*1000,
                'hh_load': topMW*1000 - non_hh_load,
                'non_hh_load': non_hh_load,
                'max_trafo_load': max_trafo_load,
                'base_trafo_count': base_trafo_count,
                'scen_#_of_crit_transformers': max_crit_trafo,
                'scen_#_of_transformers': len(net.trafo),
                'base_km_of_lines': base_line_km,
                'scen_km_of_crit_lines': max_crit_lines_km,
                'scen_new_km_of_lines': net.line["length_km"].sum()
            }

            hh_attribution = scenarios[scen_idx][tariff]["HH_attribution"]

            scenario_results = {
                "reinforcement_result": reinforcement_result,  # singular name
                "hh_attribution": hh_attribution
            }

            # Iteratively add to nested results
            results.setdefault(scen_idx, {})[tariff] = scenario_results

    _save_results(results, grid_code, seed)