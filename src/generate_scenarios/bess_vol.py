from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy import sparse
from scipy.optimize import linprog
from src.generate_scenarios.bess_lp_common import ensure_pv_positive, soc_blocks, terminal_row

# ---------- constants ----------
ROUNDTRIP_EFF_DEFAULT = 0.89
MAX_CHARGE_KW_DEFAULT = 11.5
MAX_DISCHARGE_KW_DEFAULT = 11.5

# Avoid thread oversubscription when parallelizing across buses
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------- helpers -------------------------
def _coalesce_caps(caps):
    if caps is None:
        return 0.0, 0.0, 0.0, 1.0, 1.0
    if isinstance(caps, (int, float, np.floating)) or np.isscalar(caps):
        e = float(caps)
        if not np.isfinite(e) or e <= 0.0:
            return 0.0, 0.0, 0.0, 1.0, 1.0
        eta_split = float(np.sqrt(ROUNDTRIP_EFF_DEFAULT))
        return e, MAX_CHARGE_KW_DEFAULT, MAX_DISCHARGE_KW_DEFAULT, eta_split, eta_split
    if isinstance(caps, dict):
        E = float(caps.get("E_kWh", caps.get("E", 0.0)))
        if E <= 0.0:
            return 0.0, 0.0, 0.0, 1.0, 1.0
        Pch  = float(caps.get("P_ch_max_kW", caps.get("P_max_kW", MAX_CHARGE_KW_DEFAULT)))
        Pdis = float(caps.get("P_dis_max_kW", caps.get("P_max_kW", MAX_DISCHARGE_KW_DEFAULT)))
        eta_rt = float(caps.get("eta_rt", ROUNDTRIP_EFF_DEFAULT))
        eta_c = float(caps.get("eta_c", np.sqrt(eta_rt)))
        eta_d = float(caps.get("eta_d", np.sqrt(eta_rt)))
        return E, Pch, Pdis, eta_c, eta_d
    return 0.0, 0.0, 0.0, 1.0, 1.0

def _select_time_window(df: pd.DataFrame,
                        time_col: str = "time",
                        window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> np.ndarray:
    if window is None:
        return np.ones(len(df), dtype=bool)
    start, end = window
    t = pd.to_datetime(df[time_col])
    return (t >= start) & (t <= end)

class _Triplet:
    """Minimal COO triplet builder."""
    def __init__(self):
        self.r=[]; self.c=[]; self.v=[]
    def add(self, i, j, val):
        self.r.append(i); self.c.append(j); self.v.append(val)
    def coo(self, shape):
        return sparse.coo_matrix((self.v, (self.r, self.c)), shape=shape, dtype=np.float64)

# ---------- per-bus worker ----------
def _solve_vol_one_bus(
    *, bus_id, time_values, demand_kW, pv_gen_kW,
    delta_h, caps, solver_method, epsilon_spill,
    beta_throughput_per_kWh, prefer_monthly_cyclic, initial_soc_frac,
    price_import_per_kWh: float = 1.0,   # NEW: flat volumetric price (€/kWh)
):
    E, Pch, Pdis, eta_c, eta_d = _coalesce_caps(caps)
    T = int(len(demand_kW))

    # degenerate
    if E <= 0.0 or (Pch <= 0.0 and Pdis <= 0.0) or T == 0:
        return {"bus_id": bus_id, "ok": True, "msg": "no work",
                "P_BESS": np.zeros(T, np.float32),
                "soc_times": time_values, "soc_kWh": np.zeros(T, np.float32)}

    # normalize inputs
    D  = np.asarray(demand_kW, dtype=np.float64)
    PV = ensure_pv_positive(np.asarray(pv_gen_kW, dtype=np.float64),
                            pv_positive_is_generation=True)  # <- we pass PV≥0 here

    # layout
    n_c = n_d = n_gp = n_sp = T
    n_s = T + 1
    o_c  = 0
    o_d  = o_c + n_c
    o_s  = o_d + n_d
    o_gp = o_s + n_s
    o_sp = o_gp + n_gp
    n_vars = o_sp + n_sp
    sl = lambda o, L: slice(o, o+L)

    # objective: min sum_t p * gp_t * Δt  (+ tiny penalties if requested)
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[sl(o_gp, n_gp)] = price_import_per_kWh * delta_h
    if epsilon_spill > 0:
        c_obj[sl(o_sp, n_sp)] = epsilon_spill
    if beta_throughput_per_kWh > 0:
        c_obj[sl(o_c, n_c)] += beta_throughput_per_kWh * delta_h
        c_obj[sl(o_d, n_d)] += beta_throughput_per_kWh * delta_h

    # --- build A_eq and b_eq ---
    I_T = sparse.eye(T, format="csr")
    Z   = lambda m, n: sparse.csr_matrix((m, n), dtype=np.float64)

    # (1) Power balance:  c - d - gp + spill = -D + PV
    Aeq_bal = sparse.hstack([
        I_T,           # c
        -I_T,          # d
        Z(T, T+1),     # s
        -I_T,          # gp
        I_T,           # spill
    ], format="csr")
    beq_bal = (-D + PV)

    # (2) SoC dynamics via shared blocks
    Ac, Ad, Sdiff = soc_blocks(T, delta_h, eta_c, eta_d)
    Aeq_soc = sparse.hstack([Ac, Ad, Sdiff, Z(T, T), Z(T, T)], format="csr")
    beq_soc = np.zeros(T, dtype=np.float64)

    # (3) Terminal SoC row via shared builder
    s0_val = float(initial_soc_frac * E)
    row_s, rhs_s, s0_bounds = terminal_row(T, prefer_monthly_cyclic, s0_val)
    Aeq_term = sparse.hstack([Z(1, T), Z(1, T), row_s, Z(1, T), Z(1, T)], format="csr")
    beq_term = rhs_s

    A_eq = sparse.vstack([Aeq_bal, Aeq_soc, Aeq_term], format="csr")
    b_eq = np.concatenate([beq_bal, beq_soc, beq_term]).astype(np.float64)

    # inequalities: none
    A_ub = None; b_ub = None

    # bounds
    bounds = []
    bounds.extend((0.0, Pch)  for _ in range(n_c))       # c
    bounds.extend((0.0, Pdis) for _ in range(n_d))       # d
    # s[0] bounds come from terminal_row:
    bounds.append(s0_bounds)                              # s[0]
    bounds.extend((0.0, E)        for _ in range(T))      # s[1..T]
    bounds.extend((0.0, None)     for _ in range(n_gp))   # gp (imports)
    bounds.extend((0.0, None)     for _ in range(n_sp))   # spill (exports)

    # solve
    res = linprog(
        c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method=solver_method,
        options={"presolve": True, "dual_feasibility_tolerance": 1e-9},
    )
    if not res.success:
        return {"bus_id": bus_id, "ok": False, "msg": f"LP failed: {res.status} {res.message}"}

    x = res.x
    c = x[sl(o_c, n_c)]
    d = x[sl(o_d, n_d)]
    s = x[sl(o_s, n_s)]

    return {
        "bus_id": bus_id, "ok": True, "msg": "ok",
        "P_BESS": (c - d).astype(np.float32),
        "soc_times": time_values,
        "soc_kWh": s[:-1].astype(np.float32)
    }

# ---------- main parallel VOL ----------
def bess_vol_parallel(
    bus_df_scn: pd.DataFrame,
    *,
    cap_by_bus: pd.Series | dict,
    delta_h: float = 0.25,
    solver_method: str = "highs",
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    epsilon_spill: float = 1e-6,
    beta_throughput_per_kWh: float = 0.0,
    prefer_monthly_cyclic: bool = True,
    initial_soc_frac: float = 0.25,
    price_import_per_kWh: float = 1.0,

    verbose: bool = False,
    log_every: int = 5,
    n_workers: int = 10,
    use_processes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    t0 = time.perf_counter()
    need_cols = {"bus_id", "time", "P_HH_HP_EV", "P_PV"}
    missing = need_cols - set(bus_df_scn.columns)
    if missing:
        raise ValueError(f"bus_df_scn missing required columns: {missing}")

    df = bus_df_scn.copy()
    df["time"] = pd.to_datetime(df["time"])
    mask = _select_time_window(df, "time", time_window)
    sub = df.loc[mask].copy()

    # empty-window => return an empty, window-shaped result (not the full df)
    if sub.empty:
        if verbose:
            print("[VOL|PAR] Empty window.")
        return df.loc[[]].assign(P_BESS=0.0), pd.DataFrame(columns=["bus_id","time","SoC_kWh"])

    # build the output frame **from the window only**
    out = sub.copy()
    out["P_BESS"] = 0.0
    soc_records = []

    groups = list(sub.groupby("bus_id", sort=True))
    n_buses = len(groups)
    if verbose:
        w0, w1 = (df["time"].min(), df["time"].max()) if time_window is None else time_window
        print(f"[VOL|PAR] {n_buses} buses over {w0} → {w1}")

    # choose backend
    n_workers = n_workers or max(1, os.cpu_count() or 1)
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    futures = []
    skipped = 0
    with Executor(max_workers=n_workers) as ex:
        for i, (bus_id, g) in enumerate(groups, start=1):
            idx   = g.index.to_numpy()
            t_vals= g["time"].to_numpy()
            D     = g["P_HH_HP_EV"].to_numpy(float)
            PV = ensure_pv_positive(g["P_PV"].to_numpy(float),
                                    pv_positive_is_generation=False)
            caps  = cap_by_bus.get(bus_id, {}) if isinstance(cap_by_bus, (pd.Series, dict)) else {}

            # quick capacity check to avoid spawning useless workers
            E, Pch, Pdis, *_ = _coalesce_caps(caps)
            if E <= 0.0 or (Pch <= 0.0 and Pdis <= 0.0):
                out.loc[idx, "P_BESS"] = 0.0
                soc_records.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": pd.to_datetime(t_vals),
                    "SoC_kWh": np.zeros(len(t_vals), np.float32)
                }))
                skipped += 1
                if verbose and (i == 1 or i % log_every == 0):
                    print(f"[VOL|PAR] bus {bus_id}: no BESS → skipped worker")
                continue

            fut = ex.submit(
                _solve_vol_one_bus,
                bus_id=bus_id, time_values=t_vals, demand_kW=D, pv_gen_kW=PV,
                delta_h=delta_h, caps=caps, solver_method=solver_method,
                epsilon_spill=epsilon_spill, beta_throughput_per_kWh=beta_throughput_per_kWh,
                prefer_monthly_cyclic=prefer_monthly_cyclic, initial_soc_frac=initial_soc_frac,
                price_import_per_kWh=price_import_per_kWh,   # pass through
            )
            futures.append((bus_id, idx, fut))

        # collect
        for j, (bus_id, idx, fut) in enumerate(futures, start=1):
            res = fut.result()
            if not res.get("ok", False):
                raise RuntimeError(f"[VOL] LP failed for bus {bus_id}: {res.get('msg')}")
            out.loc[idx, "P_BESS"] = res["P_BESS"]
            soc_records.append(pd.DataFrame({
                "bus_id": bus_id,
                "time": pd.to_datetime(res["soc_times"]),
                "SoC_kWh": res["soc_kWh"],
            }))
            if verbose and (j == 1 or j % log_every == 0):
                print(f"[VOL|PAR] finished {j}/{len(futures)} workers")

    soc_df = (pd.concat(soc_records, ignore_index=True)
              if soc_records else pd.DataFrame(columns=["bus_id","time","SoC_kWh"]))

    if verbose:
        print(f"[VOL|PAR] Done in {time.perf_counter()-t0:.2f}s | workers={n_workers} | "
              f"backend={'proc' if use_processes else 'threads'} | skipped={skipped}")

    return out, soc_df


def bess_vol_auto_window(
    bus_df_scn: pd.DataFrame,
    *,
    cap_by_bus: pd.Series | dict,
    delta_h: float = 0.25,
    solver_method: str = "highs",
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    epsilon_spill: float = 1e-6,
    beta_throughput_per_kWh: float = 0.0,
    prefer_monthly_cyclic: bool = True,
    initial_soc_frac: float = 0.5,
    verbose: bool = False,
    log_every: int = 5,
    n_workers: int = 10,
    use_processes: bool = True,
    # new flag: when True, split a multi-month window into calendar months
    auto_split_months: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If `time_window` covers a single calendar month => run once.
    If it spans multiple calendar months and `auto_split_months=True` => run per month and concat.
    Otherwise => run once on the whole window.
    """
    # Normalize and prefilter just like your core does
    df = bus_df_scn.copy()
    df["time"] = pd.to_datetime(df["time"])
    mask = _select_time_window(df, "time", time_window)
    sub = df.loc[mask].copy()
    if sub.empty:
        if verbose: print("[VOL|AUTO] Empty window.")
        # fall back to core function for consistent empty handling
        return bess_vol_parallel(
            bus_df_scn, cap_by_bus=cap_by_bus, delta_h=delta_h, solver_method=solver_method,
            time_window=time_window, epsilon_spill=epsilon_spill,
            beta_throughput_per_kWh=beta_throughput_per_kWh,
            prefer_monthly_cyclic=prefer_monthly_cyclic, initial_soc_frac=initial_soc_frac,
            verbose=verbose, log_every=log_every, n_workers=n_workers, use_processes=use_processes
        )

    # How many calendar months are in the filtered data?
    n_months = sub["time"].dt.to_period("M").nunique()

    # If single month, or we don't want to split, just run once
    if (not auto_split_months) or (n_months == 1):
        return bess_vol_parallel(
            bus_df_scn, cap_by_bus=cap_by_bus, delta_h=delta_h, solver_method=solver_method,
            time_window=time_window, epsilon_spill=epsilon_spill,
            beta_throughput_per_kWh=beta_throughput_per_kWh,
            prefer_monthly_cyclic=prefer_monthly_cyclic, initial_soc_frac=initial_soc_frac,
            verbose=verbose, log_every=log_every, n_workers=n_workers, use_processes=use_processes
        )

    # Split by calendar month and run per month
    if verbose:
        tw_str = f"{sub['time'].min()} → {sub['time'].max()}"
        print(f"[VOL|AUTO] Window spans {n_months} months | {tw_str} | splitting by month")

    outs = []
    socs = []
    for per, g in sub.groupby(sub["time"].dt.to_period("M"), sort=True):
        m_start = g["time"].min()
        m_end   = g["time"].max()
        if verbose:
            print(f"[VOL|AUTO] Month {per}: {m_start} → {m_end}")

        out_m, soc_m = bess_vol_parallel(
            bus_df_scn,
            cap_by_bus=cap_by_bus,
            delta_h=delta_h,
            solver_method=solver_method,
            time_window=(m_start, m_end),
            epsilon_spill=epsilon_spill,
            beta_throughput_per_kWh=beta_throughput_per_kWh,
            # For monthly subproblems, cyclic usually makes most sense:
            prefer_monthly_cyclic=prefer_monthly_cyclic,
            initial_soc_frac=initial_soc_frac,
            verbose=verbose,
            log_every=log_every,
            n_workers=n_workers,
            use_processes=use_processes,
        )

        out_m = out_m.copy()
        out_m["month"] = str(per)
        soc_m = soc_m.copy()
        soc_m["month"] = str(per)

        outs.append(out_m)
        socs.append(soc_m)

    out_all = (pd.concat(outs, ignore_index=True)
             .sort_values(["bus_id", "time"])
             .drop(columns=["month"], errors="ignore"))
    soc_all = (pd.concat(socs, ignore_index=True)
                .sort_values(["bus_id", "time"])
                .drop(columns=["month"], errors="ignore"))
    return out_all, soc_all
