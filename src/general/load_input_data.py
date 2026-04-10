from pathlib import Path
import pandas as pd

# Load all required input data for the simulation - Data loaded:

# Base load data
# - load_data.csv: Load data for households and heat pumps in Hamelin 2019, 15min intervals
#     - Source: Schlemminger et al., 2022, https://doi.org/10.1038/s41597-022-01156-1
# - household_info.csv: Metadata for sampled households
#     - Source: Schlemminger et al., 2022, https://doi.org/10.1038/s41597-022-01156-1

# EV data
# - trips.csv: Trips dataset from 2019
#     - Source: Karlsruher Institut für Technologie. Deutsches Mobilitätspanel. 2020, URL https://mobilitaetspanel.ifv.kit.edu/index.php
# - ev_types.csv: 2024 top-selling EVs in Germany by category, with specs
#     - Source: Kraftfahrbundesamt Germany, Neuzulassungen 2024, https://www.kba.de/DE/Statistik/Produktkatalog/produkte/Fahrzeuge/fz10/fz10_gentab.html
# - temperature.csv: Temperature data for Hamelin 2019
#     - Source: Schlemminger et al., 2022, https://doi.org/10.1038/s41597-022-01156-1

# PV data
# - roof_size.csv: Roof size, roof angle 
#     - Source: Mainzer et al., 2014, https://doi.org/10.1016/j.solener.2014.04.015
# - pv_production.csv: PV production for 2019 per m2 hour in kW, for 3 different angles 
#     - Source: Renewables.ninja, Pfenninger & Staffell, 2016, https://doi.org/10.1016/j.energy.2016.08.060
#         - Data for Hamelin 2019 - as in Schlemminger dataset
#         - Basis of 200W/m2 (20% Wirkungsgrad) as used by 1Komma5 Grad, etc.

# Building probability info
# - units_per_house_probs.csv: Average distribution of household size per area type in Germany
#     - Source: Zensus 2022 - Größe des privaten Haushalts (bis 4 und mehr Pes.) - Code: 5000H-1002, https://ergebnisse.zensus2022.de/datenbank/online/statistic/5000H/table/5000H-1002
# - people_per_unit_probs.csv: Average distribution of households per building and area type in Germany
#     - Source: Zensus 2022 - Wohnungen im Gebäude - Code: 3000G-1012, https://ergebnisse.zensus2022.de/datenbank/online/statistic/3000G/table/3000G-1012
# - cars_per_household_probs.csv: Number of cars per household by region and household size
#     - Source: Karlsruher Institut für Technologie. Deutsches Mobilitätspanel. 2020, URL https://mobilitaetspanel.ifv.kit.edu/index.php
# - private_parking_probs.csv: Probability of private parking spot per car in household - if yes, assumed availability of charging at home in the long run
#     - Source: Karlsruher Institut für Technologie. Deutsches Mobilitätspanel. 2020, URL https://mobilitaetspanel.ifv.kit.edu/index.php

def load_input_data():
    """Load all required input data safely with absolute paths."""

    # --- Resolve project root ---
    # Example:  src/general/load_input_data.py  →  project_root = .../v5
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    # --- Household input data ---
    load_data = pd.read_csv(data_dir / "1_load_profiles" / "load_data.csv", sep=",")
    household_info = pd.read_csv(data_dir / "1_load_profiles" / "household_info.csv", sep=",")

    # --- EV input data ---
    trips = pd.read_csv(data_dir / "2_ev_input" / "trips.csv", sep=",")
    ev_types = pd.read_csv(data_dir / "2_ev_input" / "ev_types.csv", sep=",")
    temperature = pd.read_csv(data_dir / "2_ev_input" / "2019_temperature.csv", sep=",")

    # --- PV input data ---
    roof_size = pd.read_csv(data_dir / "3_pv_input" / "roof_size.csv", sep=",")
    pv_production = pd.read_csv(data_dir / "3_pv_input" / "pv_production.csv", sep=",")

    # --- Probability information ---
    probs_dir = data_dir / "4_probabilities"
    units_per_house_probs = pd.read_csv(probs_dir / "units_per_house_probs.csv", sep=",", index_col=0)
    people_per_unit_probs = pd.read_csv(probs_dir / "people_per_unit_probs.csv", sep=",", index_col=0)
    cars_per_household_probs = pd.read_csv(probs_dir / "cars_per_household_probs.csv", sep=",", index_col=0)
    private_parking_probs = pd.read_csv(probs_dir / "private_parking_probs.csv", sep=",", index_col=0)

    return (
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
    )