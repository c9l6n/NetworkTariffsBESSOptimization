import numpy as np
import pandas as pd
import random

# ----------------------------
# Assumptions made:
# 1. Probability for a flat roof is 9% in Germany
# 2. PV system size is 0.2 kWp per m² of roof area
# 3. BESS size based on PV system size, we use 1.5 kWh/kWp
#     - Triangulation from:
#       - https://solar.htw-berlin.de/publikationen/auslegung-von-solarstromspeichern/ - here <1.5 kWh/kWp
#       - https://doi.org/10.1016/j.est.2023.107299 - here ~1 kWh/kWp, but with future growth assumptions
#       - Enpal and 1komma5grad recommendations for German sizing: 1-1.5 kWh/kWp (2025) 
#          - https://www.enpal.de/stromspeicher/kosten
#          - https://1komma5.com/de/stromspeicher/groesse-berechnen/
# ----------------------------


# ----------------------------
# Helper functions
# ---------------------------- 

def get_grid_type(grid_code):
    if 'urban' in grid_code:
        return 'urban'
    elif 'semiurb' in grid_code:
        return 'semiurb'
    elif 'rural' in grid_code:
        return 'rural'
    elif 'comm' in grid_code:
        return 'semiurb'
    elif 'mixed' in grid_code:
        return 'semiurb'
    else:
        raise ValueError("Unknown grid type in grid code.")

def sample_units(cat):
    if cat == '1_unit':
        return 1
    elif cat == '2_units':
        return 2
    elif cat == '3_6_units':
        return np.random.randint(3, 7)
    elif cat == '7_12_units':
        return np.random.randint(7, 13)
    elif cat == '13+_units':
        return np.random.randint(13, 21)

def sample_people(cat):
    if cat == '1_person':
        return 1
    elif cat == '2_persons':
        return 2
    elif cat == '3_persons':
        return 3
    elif cat == '4+_persons':
        return 4

def categorize_n_units(n_units):
    if n_units == 1:
        return 1
    elif n_units == 2:
        return 2
    else:
        return 3
    
def assign_pv_bess_properties(n_units_category, roof_size, seed=42): 
    random.seed(seed)
    p_flat = 0.09 # probability of flat roof is 9%
    is_flat = random.random() < p_flat
    roof_type = 'flat' if is_flat else 'slanted'

    match = roof_size[roof_size['building_units'] == n_units_category]
    if match.empty:
        return pd.Series([0, 0, 'none'])

    if roof_type == 'flat':
        pv_size_kWp = match['flat_roof_solar_m2'].values[0] * 0.2
        pv_angle = int(match['flat_roof_angle'].values[0] * 100)
    else:
        pv_size_kWp = match['slanted_roof_solar_m2'].values[0] * 0.2
        pv_angle = int(match['slanted_roof_angle'].values[0] * 100)
    
    bess_size_kWh = pv_size_kWp * 1.5  # 1.5 kWh per kWp of PV size

    return roof_type, pv_size_kWp, pv_angle, bess_size_kWh
    
# ----------------------------
# Generate dataframe to input data
# ----------------------------

import numpy as np
import pandas as pd

def generate_household_dataframe(
    distinct_buses,
    grid_type,
    units_per_house_probs,
    people_per_unit_probs,
    cars_per_household_probs,
    private_parking_probs,
    roof_size,
    seed=None
):
    """
    Returns a DataFrame with one row per household unit across all buses.
    Expects helper functions:
      - sample_units(unit_category) -> int
      - sample_people(people_category) -> int
      - categorize_n_units(n_units) -> str/category
      - assign_pv_bess_properties(n_units_category, roof_size, seed=None)
          -> (roof_type, pv_size_kWp, pv_angle, bess_size_kWh)
    """
    if seed is not None:
        np.random.seed(seed)

    all_rows = []
    hh_counter = 0

    # We will sample categories by their column labels (not raw .values),
    # which is safer and keeps alignment intuitive.
    unit_labels = units_per_house_probs.columns
    people_labels = people_per_unit_probs.columns

    # Pull and normalize probability vectors once per grid_type
    try:
        unit_probs = units_per_house_probs.loc[grid_type]
    except KeyError:
        raise KeyError(f"grid_type {grid_type!r} not found in units_per_house_probs index")

    unit_probs = unit_probs / unit_probs.sum()

    try:
        base_people_probs = people_per_unit_probs.loc[grid_type]
    except KeyError:
        raise KeyError(f"grid_type {grid_type!r} not found in people_per_unit_probs index")

    base_people_probs = base_people_probs / base_people_probs.sum()

    for bus_id in distinct_buses:
        bus_rows = []

        # Sample number of units for this bus
        unit_cat = np.random.choice(unit_labels, p=unit_probs.to_numpy())
        n_units = sample_units(unit_cat)

        # Generate one household row per unit
        for _ in range(int(n_units)):
            people_cat = np.random.choice(people_labels, p=base_people_probs.to_numpy())
            n_people = sample_people(people_cat)

            # Cars and parking determined by people category
            try:
                expected_cars = float(cars_per_household_probs.loc[grid_type, people_cat])
            except KeyError:
                raise KeyError(
                    f"Missing cars_per_household_probs for grid_type={grid_type!r}, people_cat={people_cat!r}"
                )
            n_cars = np.random.poisson(max(expected_cars, 0.0))

            try:
                parking_prob = float(private_parking_probs.loc[grid_type, people_cat])
            except KeyError:
                raise KeyError(
                    f"Missing private_parking_probs for grid_type={grid_type!r}, people_cat={people_cat!r}"
                )
            parking_prob = min(max(parking_prob, 0.0), 1.0)  # clamp to [0,1]
            n_parking = np.random.binomial(n=int(n_cars), p=parking_prob)

            bus_rows.append(
                {
                    "bus_id": bus_id,
                    "hh_id": f"{hh_counter}",
                    "num_people": int(n_people),
                    "num_cars": int(n_cars),
                    "num_parking_spots": int(n_parking),
                }
            )
            hh_counter += 1

        # Create PV and BESS columns (same across all rows for this bus)
        n_units_category = categorize_n_units(n_units)
        roof_type, pv_size_kWp, pv_angle, bess_size_kWh = assign_pv_bess_properties(
            n_units_category, roof_size, seed
        )

        bus_df = pd.DataFrame(bus_rows)
        if not bus_df.empty:
            bus_df["roof_type"] = roof_type
            bus_df["pv_size_kWp"] = pv_size_kWp
            bus_df["pv_angle"] = pv_angle
            bus_df["bess_size_kWh"] = bess_size_kWh

        all_rows.append(bus_df)

    # Properly combine per-bus DataFrames
    if len(all_rows) == 0:
        return pd.DataFrame(
            columns=[
                "bus_id",
                "hh_id",
                "num_people",
                "num_cars",
                "num_parking_spots",
                "roof_type",
                "pv_size_kWp",
                "pv_angle",
                "bess_size_kWh",
            ]
        )

    return pd.concat(all_rows, ignore_index=True)

# ----------------------------
# Match households for easier access
# ----------------------------

def match_households(household_profiles, household_info):
    # Pre-group household_info by number of inhabitants for faster access
    grouped_info = household_info.groupby('Number of inhabitants')

    def get_random_household(num_people):
        if num_people in grouped_info.groups:
            return grouped_info.get_group(num_people).sample(n=1).iloc[0]
        else:
            # Fallback strategy: sample from closest available group
            closest_group = min(grouped_info.groups.keys(), key=lambda x: abs(x - num_people))
            print(f"[Warning] No exact match for {num_people} people. Using closest: {closest_group}")
            return grouped_info.get_group(closest_group).sample(n=1).iloc[0]

    # Vectorized-like apply (still row-wise, but faster via pre-grouping)
    matched = household_profiles['num_people'].apply(get_random_household)
    
    # Merge results
    household_profiles = household_profiles.reset_index(drop=True)
    household_profiles['input_hh_id'] = matched['household id'].values

    return household_profiles

# ----------------------------
# Main generate households function
# ----------------------------

def generate_households(net, grid_code, household_info, 
                        units_per_house_probs, people_per_unit_probs, 
                        cars_per_household_probs, private_parking_probs,
                        roof_size, seed=None):
    """
    Generate load profiles for a given grid code and percentages of heat pumps, EVs, and PV systems.
    """
    # Filter net.load for rows where the "profile" column starts with "H"
    filtered_loads = net.load[net.load['profile'].str.startswith('H', na=False)]

    # Get distinct bus IDs
    distinct_buses = filtered_loads['bus'].unique()

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Get grid type
    grid_type = get_grid_type(grid_code)

    # Generate household profiles
    household_df = generate_household_dataframe(
        distinct_buses,
        grid_type,
        units_per_house_probs,
        people_per_unit_probs,
        cars_per_household_probs,
        private_parking_probs,
        roof_size,
        seed)
    
    # Match household profiles with household info
    household_df = match_households(household_df, household_info)

    return household_df