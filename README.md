# Network Tariff BESS Optimization Grid Reinforcement Simulation

This repository contains a modular simulation framework to assess the impact of different network tariffs on electricity distribution grids. It supports:

- Scenario generation (e.g. BESS, EV/HP load evolution, probabilistic inputs)
- Simulation of BESS Storage Optimization based on underlying network tariff
- Grid reinforcement simulations across LV, MV, and HV networks

---

## 📁 Repository Structure

   ```bash
  project-root/
│
├── main.py                      # Main CLI entry point (scenarios & reinforcement)
│
├── data/                        # Input data & generated scenario files
│   ├── 1_load_profiles/
│   │   ├── 2019_data_15min.hdf5 # File to be downloaded here: https://doi.org/10.5281/zenodo.5642902, file 2019_data_15min.hdf5
│   │   ├── household_info.csv
│   │   └── load_data.csv
│   │
│   ├── 2_ev_input/
│   │   ├── 2019_temperature.csv
│   │   ├── ev_types.csv
│   │   └── trips.csv
│   │
│   ├── 3_pv_input/
│   │   ├── pv_production.csv
│   │   └── roof_size.csv
│   │
│   ├── 4_probabilities/
│   │   ├── cars_per_household_probs.csv
│   │   ├── people_per_unit_probs.csv
│   │   ├── private_parking_probs.csv
│   │   └── units_per_house_probs.csv
│   │
│   ├── 5_grid_input/
│   │   ├── standardLines.csv
│   │   └── standardTrafos.csv
│   │
│   ├── reinforcement_files/     # Generated reinforcement outputs
│   └── scenario_files_LV/       # Generated scenario files (LV)
│
├── src/                         # Core model logic
│   ├── analyze_results/
│   │   ├── analyze_results_cost.py
│   │   └── prepare_results.py
│   │
│   ├── general/
│   │   └── load_input_data.py
│   │
│   ├── generate_scenarios/      # Scenario generation pipeline
│   │   ├── generate_scenarios.py
│   │   ├── generate_households.py
│   │   ├── generate_loads.py
│   │   ├── setup_scenarios.py
│   │   ├── calculate_bess.py
│   │   ├── bess_*.py
│   │
│   ├── run_reinforcement/       # Reinforcement simulation runners
│   │   ├── run_lv_reinforcement_scenario.py
│   │   ├── run_mv_reinforcement_scenario.py
│   │   ├── run_hv_reinforcement_scenario.py
│   │   └── reinforce_grid.py

```
---

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy simbench pandapower h5py

2. **Execute the Full Simulation (e.g., with seeds 41-45)**:
   ```bash
   python main.py scenarios --seeds 41-45

3. **Analyze Results**:

   Use notebook in ipynb/ to generate results data as represented in paper.
  
---

## 📊 Output

The model will generate:
- CSV summaries per grid archetype in /data/results/
- Optional plots and tables via notebooks
