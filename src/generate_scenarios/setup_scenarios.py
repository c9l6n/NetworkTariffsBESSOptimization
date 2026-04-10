import numpy as np
import pandas as pd
from typing import Tuple

import numpy as np
import pandas as pd
from itertools import product
from typing import List, Tuple, Dict, Iterable, Optional

def _assert_unique(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    """Fail fast if key columns are not unique."""
    dup = df.duplicated(list(cols)).any()
    if dup:
        raise ValueError(f"{name}: duplicate keys in {cols}")

def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], utc=False)

def _uniq_sorted(vals: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vals), dtype=float)
    return np.unique(arr)

def _factorize_hh_and_groups(base: pd.DataFrame, household_df: pd.DataFrame):
    """Factorize household IDs and (bus_id, time) group keys for fast aggregation."""
    hh_uniques = household_df['hh_id'].unique()
    n_hh = hh_uniques.size
    hh_codes = pd.Categorical(base['hh_id'], categories=hh_uniques).codes
    if (hh_codes < 0).any():
        missing = base.loc[hh_codes < 0, 'hh_id'].unique()
        raise ValueError(f"Found unknown hh_id in base: {missing[:5]}{'...' if len(missing)>5 else ''}")

    bus_uniques = household_df['bus_id'].unique()
    n_bus = bus_uniques.size

    grp_keys = pd.MultiIndex.from_arrays([base['bus_id'].to_numpy(), base['time'].to_numpy()])
    grp_ids, grp_index = pd.factorize(grp_keys, sort=False)
    n_groups = grp_index.size
    grp_bus_ids  = grp_index.get_level_values(0).to_numpy()
    grp_time_vals = grp_index.get_level_values(1).to_numpy()
    grp_bus_codes = pd.Categorical(grp_bus_ids, categories=bus_uniques).codes

    return {
        "hh_uniques": hh_uniques, "n_hh": n_hh, "hh_codes": hh_codes,
        "bus_uniques": bus_uniques, "n_bus": n_bus,
        "grp_index": grp_index, "grp_ids": grp_ids, "n_groups": n_groups,
        "grp_bus_ids": grp_bus_ids, "grp_time_vals": grp_time_vals, "grp_bus_codes": grp_bus_codes
    }

def _align_pv_to_groups(pv_load_df: pd.DataFrame, grp_index: pd.MultiIndex, n_groups: int) -> np.ndarray:
    """Map PV (bus_id,time) onto group indices and sum to a dense array."""
    # Validate keys & dtype
    _assert_unique(pv_load_df, ['bus_id', 'time'], "pv_load_df")
    _ensure_datetime(pv_load_df, 'time')

    pv_keys = pd.MultiIndex.from_arrays([pv_load_df['bus_id'].to_numpy(),
                                         pv_load_df['time'].to_numpy()])
    # Use MultiIndex.get_indexer for speed/correctness
    pv_to_grp = grp_index.get_indexer(pv_keys)
    PV_by_group_base = np.zeros(n_groups, dtype=np.float32)
    valid = pv_to_grp >= 0
    if valid.any():
        np.add.at(PV_by_group_base, pv_to_grp[valid],
                  pv_load_df.loc[valid, 'P_PV'].to_numpy(dtype=np.float32, copy=False))
    return PV_by_group_base

def _prepare_scenarios_and_draws(
    hp_percentages: Optional[List[float]],
    ev_percentages: Optional[List[float]],
    pv_percentages: Optional[List[float]],
    scenario_space: Optional[List[Tuple[float,float,float]]],
    n_hh: int, n_bus: int, seed: Optional[int]
):
    """Build scenario lists and monotone adoption matrices with independent draws per DER."""
    rng = np.random.default_rng(seed)

    if scenario_space is None:
        if hp_percentages is None or ev_percentages is None or pv_percentages is None:
            raise ValueError("Either scenario_space OR all of (hp_percentages, ev_percentages, pv_percentages) must be provided.")
        hp_vals = _uniq_sorted(hp_percentages)
        ev_vals = _uniq_sorted(ev_percentages)
        pv_vals = _uniq_sorted(pv_percentages)
        scenario_list = list(product(hp_vals.tolist(), ev_vals.tolist(), pv_vals.tolist()))
    else:
        hp_vals = _uniq_sorted(s[0] for s in scenario_space)
        ev_vals = _uniq_sorted(s[1] for s in scenario_space)
        pv_vals = _uniq_sorted(s[2] for s in scenario_space)
        scenario_list = scenario_space

    hp_thr = np.clip(hp_vals, 0.0, 100.0) / 100.0
    ev_thr = np.clip(ev_vals, 0.0, 100.0) / 100.0
    pv_thr = np.clip(pv_vals, 0.0, 100.0) / 100.0

    # Independent draws by technology (avoid perfect correlation)
    hh_rand_hp = rng.random((n_hh, 1), dtype=np.float64).astype(np.float32)
    hh_rand_ev = rng.random((n_hh, 1), dtype=np.float64).astype(np.float32)
    bus_rand_pv = rng.random((n_bus, 1), dtype=np.float64).astype(np.float32)

    hp_assign = (hh_rand_hp < hp_thr.reshape(1, -1))   # shape (n_hh, len(hp_thr))
    ev_assign = (hh_rand_ev < ev_thr.reshape(1, -1))   # shape (n_hh, len(ev_thr))
    pv_assign = (bus_rand_pv < pv_thr.reshape(1, -1))  # shape (n_bus, len(pv_thr))

    hp_pos = {float(v): i for i, v in enumerate(hp_vals.tolist())}
    ev_pos = {float(v): i for i, v in enumerate(ev_vals.tolist())}
    pv_pos = {float(v): i for i, v in enumerate(pv_vals.tolist())}

    return {
        "hp_vals": hp_vals, "ev_vals": ev_vals, "pv_vals": pv_vals,
        "hp_pos": hp_pos, "ev_pos": ev_pos, "pv_pos": pv_pos,
        "scenario_list": scenario_list,
        "hp_assign": hp_assign, "ev_assign": ev_assign, "pv_assign": pv_assign
    }

# --------------------------------------
# Main scenario setup function
# --------------------------------------

def build_setup(
    household_df: pd.DataFrame,
    hh_load_df: pd.DataFrame,
    hp_load_df: pd.DataFrame,
    ev_load_df: pd.DataFrame,
    pv_load_df: pd.DataFrame,
    hp_percentages: Optional[List[float]] = None,
    ev_percentages: Optional[List[float]] = None,
    pv_percentages: Optional[List[float]] = None,
    scenario_space: Optional[List[Tuple[float, float, float]]] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Build the base frame, factorize IDs/groups, align PV, and prepare scenarios/draws.

    Returns
    -------
    setup : dict
        {
          "base", "P_HH", "P_HP_base", "P_EV_base",
          "PV_by_group_base",
          # plus everything returned by _factorize_hh_and_groups(...),
          # plus everything returned by _prepare_scenarios_and_draws(...)
        }
    """
    # 1) Build base frame once
    # Validate uniqueness and datetime before merges
    _assert_unique(hh_load_df, ['hh_id','time'], "hh_load_df")
    _assert_unique(hp_load_df, ['hh_id','time'], "hp_load_df")
    _assert_unique(ev_load_df, ['hh_id','time'], "ev_load_df")
    _ensure_datetime(hh_load_df, 'time')
    _ensure_datetime(hp_load_df, 'time')
    _ensure_datetime(ev_load_df, 'time')

    base = (
        hh_load_df
        .merge(hp_load_df[['hh_id','time','P_HP']], on=['hh_id','time'], how='left', validate='one_to_one')
        .merge(ev_load_df[['hh_id','time','P_EV_total']], on=['hh_id','time'], how='left', validate='one_to_one')
    )
    base[['P_HP','P_EV_total']] = base[['P_HP','P_EV_total']].fillna(0)

    # Keep numeric arrays handy
    P_HH      = base['P_HH'].to_numpy(dtype=np.float32, copy=False)
    P_HP_base = base['P_HP'].to_numpy(dtype=np.float32, copy=False)
    P_EV_base = base['P_EV_total'].to_numpy(dtype=np.float32, copy=False)

    # 2) Factorize IDs & groups
    idgrp = _factorize_hh_and_groups(base, household_df)
    grp_index = idgrp['grp_index']

    # 3) PV alignment to group space
    _ensure_datetime(pv_load_df, 'time')
    PV_by_group_base = _align_pv_to_groups(pv_load_df, grp_index, idgrp['n_groups'])
    # Note: PV_by_group_base is positive generation; you will negate on application.

    # 4) Scenario percentages & random assignments (monotone, reproducible)
    if scenario_space is None:
        scen = _prepare_scenarios_and_draws(
            hp_percentages = hp_percentages,
            ev_percentages = ev_percentages,
            pv_percentages = pv_percentages,
            scenario_space = None,
            n_hh = idgrp['n_hh'],
            n_bus = idgrp['n_bus'],
            seed = seed
        )
    else:
        scen = _prepare_scenarios_and_draws(
            hp_percentages = None,
            ev_percentages = None,
            pv_percentages = None,
            scenario_space = scenario_space,
            n_hh = idgrp['n_hh'],
            n_bus = idgrp['n_bus'],
            seed = seed
        )

    setup = {
        "base": base,
        "P_HH": P_HH, "P_HP_base": P_HP_base, "P_EV_base": P_EV_base,
        "PV_by_group_base": PV_by_group_base,
        **idgrp,
        **scen
    }
    return setup

def _normalize_scenarios(user_scenarios: list[list[int]]) -> list[tuple[int,int,int]]:
    """
    Ensure scenario list starts with (0,0,0) and ends with (100,100,100),
    and that each triple is a tuple[int,int,int].
    """
    sc = [tuple(map(int, s)) for s in user_scenarios]
    if sc[0] != (0,0,0):
        sc = [(0,0,0)] + sc
    if sc[-1] != (100,100,100):
        sc = sc + [(100,100,100)]
    return sc

def select_masks_for_scenario(
    setup: dict, hp_p: float, ev_p: float, pv_p: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-level masks for HP/EV and group-level mask for PV."""
    ihp = setup['hp_pos'][float(hp_p)]
    iev = setup['ev_pos'][float(ev_p)]
    ipv = setup['pv_pos'][float(pv_p)]

    # Per-row (household) masks: shape = len(base)
    hp_mask_rows = setup['hp_assign'][:, ihp][setup['hh_codes']]
    ev_mask_rows = setup['ev_assign'][:, iev][setup['hh_codes']]

    # Per-group (bus,time) PV mask: first per-bus, then expand to groups
    pv_bus_mask = setup['pv_assign'][:, ipv]                       # shape (n_bus,)
    pv_group_mask = pv_bus_mask[setup['grp_bus_codes']]            # shape (n_groups,)

    return hp_mask_rows, ev_mask_rows, pv_group_mask.astype(bool), pv_bus_mask.astype(bool)

def materialize_scenario_frames(
    setup: dict, hp_p: float, ev_p: float, pv_p: float, household_df: pd.DataFrame, include_pv: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Build per-HH frame (bus_id, hh_id, time, P_HH, P_HP, P_EV) and
    bus-level frame (bus_id, time, P_HH_HP_EV[, P_PV]) for a given scenario.
    """
    hp_mask_rows, ev_mask_rows, pv_group_mask, pv_bus_mask = select_masks_for_scenario(setup, hp_p, ev_p, pv_p)

    # Scenario-masked household powers (kW) on 'base' rows
    P_HH = setup['P_HH']                                   # already float32
    P_HP = setup['P_HP_base'] * hp_mask_rows               # float32 * bool -> float32
    P_EV = setup['P_EV_base'] * ev_mask_rows               # float32 * bool -> float32
    P_total_partial = P_HH + P_HP + P_EV                   # HH+HP+EV, pre-PV/BESS

    # (a) Per-HH dataframe
    hh_df = pd.DataFrame({
        'bus_id': setup['base']['bus_id'].to_numpy(),
        'hh_id':  setup['base']['hh_id'].to_numpy(),
        'time':   setup['base']['time'].to_numpy(),
        'P_HH':   P_HH,            # kW
        'P_HP':   P_HP,            # kW
        'P_EV':   P_EV,            # kW
    })

    # (c) Bus-level aggregation via bincount to group ids
    P_par_g = np.bincount(
        setup['grp_ids'], weights=P_total_partial,
        minlength=setup['n_groups']
    ).astype(np.float32, copy=False)

    data = {
        'bus_id':       setup['grp_bus_ids'],
        'time':         setup['grp_time_vals'],
        'P_HH_HP_EV':   P_par_g,  # kW
    }

    if include_pv:
        # Apply PV adoption (per-group mask) and make generation negative
        P_PV_g = -(setup['PV_by_group_base'] * pv_group_mask.astype(np.float32))
        data['P_PV'] = P_PV_g

    bus_df = pd.DataFrame(data)

    # --- Capacity per bus (kWh) — first non-null value per bus
    cap_by_bus = (
        household_df
        .dropna(subset=['bess_size_kWh'])
        .groupby('bus_id', sort=False)['bess_size_kWh']
        .first()
    )

    mask = pd.Series(pv_bus_mask, index=cap_by_bus.index, dtype=bool)
    cap_filtered = cap_by_bus.where(mask, 0)

    return hh_df, bus_df, cap_filtered

def select_peak_month_window(bus_df_scn: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Pick the month with the highest system coincident import peak (baseline).
    Assumes: P_HH_HP_EV (kW, +), P_PV (kW, negative when generating), P_BESS present (or not yet set).
    We ignore P_BESS here to find baseline peaks.
    """
    df = bus_df_scn.copy()
    df["time"] = pd.to_datetime(df["time"])
    # baseline net at each bus (kW)
    df["P_net_baseline"] = df["P_HH_HP_EV"].astype(float) + df["P_PV"].astype(float)  # (+ import, - export)
    # system net at each timestep
    sys = df.groupby("time", as_index=False)["P_net_baseline"].sum()
    # compute monthly max
    sys["month"] = sys["time"].dt.to_period("M")
    monthly_peaks = sys.groupby("month")["P_net_baseline"].max().sort_values(ascending=False)
    peak_month = monthly_peaks.index[0]  # Period('2019-01', 'M') for example
    # window is entire month
    start = pd.Timestamp(peak_month.start_time)
    end   = pd.Timestamp(peak_month.end_time).replace(hour=23, minute=45, second=0, microsecond=0)
    return start, end

def select_topN_hours_covering_months(bus_df_scn: pd.DataFrame, N: int = 50, coverage: float = 0.8):
    """
    Optional: find the minimal number of months that cover ≥ coverage of the top-N system hours.
    Returns a list of (start, end) month windows.
    """
    df = bus_df_scn.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["P_net_baseline"] = df["P_HH_HP_EV"].astype(float) + df["P_PV"].astype(float)
    sys = df.groupby("time", as_index=False)["P_net_baseline"].sum().sort_values("P_net_baseline", ascending=False)
    top = sys.head(N).copy()
    top["month"] = top["time"].dt.to_period("M")
    counts = top["month"].value_counts()
    windows = []
    covered = 0
    for m, c in counts.sort_values(ascending=False).items():
        start = pd.Timestamp(m.start_time)
        end   = pd.Timestamp(m.end_time).replace(hour=23, minute=45, second=0, microsecond=0)
        windows.append((start, end))
        covered += c
        if covered / N >= coverage:
            break
    return windows