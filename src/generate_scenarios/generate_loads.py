import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numba import njit
import random

# ----------------------------
# Global Seed Function
# ----------------------------
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ----------------------------
# Load profile shifting function
# ----------------------------
def shift_load_profile(df, shift_minutes):
    shift_steps = int(shift_minutes / 15)
    value_col = df.columns.difference(['time'])[0]
    df_shifted = df.copy()
    df_shifted[value_col] = df[value_col].shift(shift_steps, fill_value=df[value_col].iloc[0 if shift_steps > 0 else -1])
    return df_shifted

# ----------------------------
# Household + HP Load Generation
# ----------------------------
def generate_hh_and_hp_loads(household_df, load_data, seed):
    set_global_seed(seed)
    shift_options = [-30, -15, 0, 15, 30]
    all_hh_rows = []
    all_hp_rows = []

    for _, row in household_df.iterrows():
        hh_id = row['hh_id']
        bus_id = row['bus_id']
        input_hh_id = row['input_hh_id']
        shift_minutes = np.random.choice(shift_options)

        df = load_data[load_data['hh_id'] == input_hh_id]

        df_hh = shift_load_profile(df[['time', 'P_HH']].copy(), shift_minutes)
        df_hh['bus_id'] = bus_id
        df_hh['hh_id'] = hh_id
        df_hh['time'] = pd.to_datetime(df_hh['time'])
        all_hh_rows.append(df_hh[['bus_id', 'hh_id', 'time', 'P_HH']])

        df_hp = shift_load_profile(df[['time', 'P_HP']].copy(), shift_minutes)
        df_hp['bus_id'] = bus_id
        df_hp['hh_id'] = hh_id
        df_hp['time'] = pd.to_datetime(df_hp['time'])
        all_hp_rows.append(df_hp[['bus_id', 'hh_id', 'time', 'P_HP']])

    return pd.concat(all_hh_rows, ignore_index=True), pd.concat(all_hp_rows, ignore_index=True)

# ----------------------------
# Numba-safe EV day simulation
# ----------------------------
@njit
def simulate_ev_day_jit(day_index, trips_array, battery_kWh, consumption_kWh_per_km, soc_start,
                        has_charger, week, wotag, hhid, persid, rand_array,
                        max_power_kW=11.0):
    state_array = np.ones(96, dtype=np.int32)
    distance_array = np.zeros(96, dtype=np.float32)
    soc_array = np.full(96, soc_start, dtype=np.float32)
    load_array = np.zeros(96, dtype=np.float32)
    charging = False
    soc = soc_start

    for i in range(trips_array.shape[0]):
        if trips_array[i, 3] != hhid or trips_array[i, 4] != persid:
            continue
        if trips_array[i, 5] == 0:
            continue

        ab = int(trips_array[i, 0])
        an = int(trips_array[i, 1])
        zweck = int(trips_array[i, 2])
        km = trips_array[i, 5]

        ab_min = ab % 100
        ab_hour = ab // 100
        an_min = an % 100
        an_hour = an // 100

        start_step = (ab_hour * 60 + ab_min) // 15
        end_step = (an_hour * 60 + an_min) // 15
        if end_step <= start_step:
            end_step = start_step + 1
        start_step = min(95, max(0, start_step))
        end_step = min(96, max(0, end_step))

        state_array[start_step:end_step] = 0
        per_step_dist = km / (end_step - start_step)
        distance_array[start_step:end_step] = per_step_dist

        status = 1
        if zweck in (1, 3):
            status = 2
        elif zweck != 7:
            status = 3
        if end_step < 96:
            state_array[end_step:96] = status

    for step in range(96):
        if state_array[step] == 0:
            # driving
            energy_used = distance_array[step] * consumption_kWh_per_km
            soc = max(0.0, soc - energy_used)
            soc_array[step] = soc
            charging = False
            # load_array[step] stays 0.0
        elif state_array[step] == 1:
            # at home
            # already full?
            if soc >= battery_kWh - 1e-6:
                charging = False
                soc_array[step] = soc
                # load_array[step] stays 0.0
                continue

            soc_pct = soc / battery_kWh
            if has_charger:
                prob_raw = 1.0 - 1.0 / (1.0 + np.exp(-0.15 * (100.0 * soc_pct - 60.0)))
            else:
                prob_raw = 1.0 - 1.2 / (1.0 + np.exp(-0.15 * (100.0 * soc_pct - 60.0)))
            prob = min(1.0, max(0.0, prob_raw))  # clamp to [0,1]

            if not charging and rand_array[step] < prob:
                charging = True

            if charging:
                # flat power up to max, no midnight spreading
                energy_needed = battery_kWh - soc
                if energy_needed > 0.0:
                    p_to_fill_now = energy_needed / (0.25 * 0.9)
                    power = max_power_kW if max_power_kW < p_to_fill_now else p_to_fill_now
                    energy_charged = power * 0.25 * 0.9
                    soc += energy_charged
                    if soc > battery_kWh:
                        soc = battery_kWh
                    soc_array[step] = soc
                    load_array[step] = power  # kW over this 15-min slot
                    if soc >= battery_kWh - 1e-6:
                        charging = False
                else:
                    charging = False
                    soc_array[step] = soc
            else:
                # home but not charging — record current SoC
                soc_array[step] = soc
        else:
            # away (work/other)
            charging = False
            soc_array[step] = soc
            # load_array[step] stays 0.0

    return state_array, distance_array, soc_array, load_array, soc

# ----------------------------
# EV Yearly calc
# ----------------------------

def simulate_ev_year(
    trips,
    temperature,
    battery_kWh,
    consumption_kWh_per_km,
    has_charger,
    seed=42,
    start_date="2019-01-01",
    days=365,
    warmup_days=2,                      # simulate this many days before start_date
):
    set_global_seed(seed)

    # Dates & sizes
    start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    total_days = warmup_days + days
    num_timesteps = 96 * total_days

    # Initial SoC
    soc = 0.8 * battery_kWh

    # Trip arrays
    unique_person_ids = trips[['HHID', 'PERSID']].drop_duplicates().values.tolist()
    trip_array = trips[['ABZEIT', 'ANZEIT', 'ZWECK', 'HHID', 'PERSID', 'KM']].values

    # Build a (iso_year, iso_week) → (hhid, persid) map covering the whole simulated span
    week_keys = sorted({
        ( (start_dt + timedelta(days=d)).isocalendar()[0],
          (start_dt + timedelta(days=d)).isocalendar()[1] )
        for d in range(total_days)
    })
    random.seed(seed)
    week_person_cache = {wk: random.choice(unique_person_ids) for wk in week_keys}

    # Storage
    load_full = np.zeros(num_timesteps, dtype=np.float32)
    precomputed_randoms = np.random.rand(total_days, 96).astype(np.float32)

    # Simulate
    for day in range(total_days):
        current_date = start_dt + timedelta(days=day)
        iso = current_date.isocalendar()
        week_key = (iso[0], iso[1])
        hhid, persid = week_person_cache[week_key]
        week = iso[1]
        wotag = iso[2]

        state, distance, soc_day, load, soc = simulate_ev_day_jit(
            day, trip_array, battery_kWh, consumption_kWh_per_km, soc, has_charger,
            week, wotag, hhid, persid, precomputed_randoms[day]
        )
        load_full[day * 96:(day + 1) * 96] = load

    # Build full index incl. warm-up
    time_index_full = pd.date_range(start=start_dt, periods=num_timesteps, freq="15min")
    df_full = pd.DataFrame({'time': time_index_full, 'P_EV': load_full})

    # Slice out warm-up (e.g., first 192 steps) and return exactly `days` days
    start_idx = warmup_days * 96
    end_idx = start_idx + days * 96
    df = df_full.iloc[start_idx:end_idx].reset_index(drop=True)

    return df

# ----------------------------
# EV household calc
# ----------------------------

def simulate_all_evs_for_household(household_row, trips, temperature, ev_types, seed=42):
    set_global_seed(seed)

    num_evs = int(household_row.get('num_cars', 0))
    num_chargers = int(household_row.get('num_parking_spots', 0))

    if num_evs == 0:
        time_index = pd.date_range(start="2019-01-01", periods=96 * 365, freq='15min')
        return pd.DataFrame({'time': time_index, 'P_EV_total': np.zeros(len(time_index))})

    ev_profiles = []
    for i in range(num_evs):
        has_charger = i < num_chargers
        types = ev_types['car_type'].tolist()
        probs = ev_types['probability'].tolist()
        chosen = random.choices(types, weights=probs, k=1)[0]
        ev_row = ev_types[ev_types['car_type'] == chosen].iloc[0]
        battery_kWh = ev_row['battery_kWh']
        consumption_kWh_per_km = ev_row['consumption_kWh_per_km']

        ev_df = simulate_ev_year(trips, temperature, battery_kWh, consumption_kWh_per_km, has_charger, seed=seed + i)
        ev_df.rename(columns={'P_EV': f'P_EV_{i+1}'}, inplace=True)
        ev_profiles.append(ev_df)

    merged_df = ev_profiles[0]
    for df in ev_profiles[1:]:
        merged_df = merged_df.merge(df, on='time')

    ev_cols = [col for col in merged_df.columns if col.startswith("P_EV_")]
    merged_df['P_EV_total'] = merged_df[ev_cols].sum(axis=1)
    return merged_df[['time'] + ev_cols + ['P_EV_total']]

# ----------------------------
# Main EV function
# ----------------------------

def generate_ev_loads(household_df, trips, temperature, ev_types, seed=42):
    set_global_seed(seed)
    all_results = []

    for idx, row in household_df.iterrows():
        hh_id = row['hh_id']
        bus_id = row['bus_id']

        ev_df = simulate_all_evs_for_household(
            household_row=row,
            trips=trips,
            temperature=temperature,
            ev_types=ev_types,
            seed=seed + idx
        )

        ev_df['hh_id'] = hh_id
        ev_df['bus_id'] = bus_id
        all_results.append(ev_df)

    cols = ['bus_id', 'hh_id', 'time', 'P_EV_1', 'P_EV_2', 'P_EV_3', 'P_EV_4', 'P_EV_total']

    if not all_results:
        # No households → return empty DataFrame with the expected columns
        return pd.DataFrame(columns=cols)

    result_df = pd.concat(all_results, ignore_index=True)

    # Ensure required columns exist and are ordered; any missing are created with 0
    result_df = result_df.reindex(columns=cols, fill_value=0)

    # Also convert any existing NaNs in those columns to 0 (just in case)
    return result_df.fillna(0)


# PV functions

# ----------------------------
# Main PV function
# ----------------------------

def generate_pv_loads(household_df, pv_production, seed=42):
    set_global_seed(seed)

    bus_solar_df = household_df.groupby('bus_id').agg({
        'pv_size_kWp': 'mean',
        'pv_angle': 'mean',
        'roof_type': 'first'
    }).reset_index()

    all_results = []
    for _, row in bus_solar_df.iterrows():
        angle_df = pv_production[pv_production["angle"] == row["pv_angle"]].copy()
        angle_df['P_PV'] = angle_df['electricity']

        angle_df = angle_df.drop(columns=['angle', 'electricity']).reset_index(drop=True)
        angle_df['time'] = pd.to_datetime(angle_df['time'])
        angle_df.set_index('time', inplace=True)
        angle_df = angle_df.groupby(angle_df.index).first()

        full_index = pd.date_range(start=angle_df.index.min(), end="2019-12-31 23:45:00", freq='15min')
        df_full = angle_df.reindex(full_index, method='ffill')

        df_full.reset_index(inplace=True)
        df_full.rename(columns={'index': 'time'}, inplace=True)
        df_full['P_PV'] = df_full['P_PV'] * (row['pv_size_kWp'] / 0.2)
        df_full['bus_id'] = row['bus_id']

        all_results.append(df_full)

    cols = ['bus_id', 'time', 'P_PV']
    result_df = pd.concat(all_results, ignore_index=True)
    return result_df[cols]