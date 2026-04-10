from __future__ import annotations
import os, time, hashlib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Iterable, Any, List
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

def _month_key_ser(s: pd.Series):
    return pd.to_datetime(s).dt.to_period("M")

def _clean_vol_schedule(vol_raw: pd.DataFrame, *, vol_bess_discharge_positive: bool=False) -> pd.DataFrame:
    """
    Clean VOL output:
      - unique (bus_id, time)
      - normalize to internal convention: charge-positive / discharge-negative (P_BESS = c - d)
        Pass vol_bess_discharge_positive=True if your VOL is discharge-positive to flip the sign.
    Adds a 'month' column from time for convenience.
    """
    v = vol_raw.copy()
    v["time"] = pd.to_datetime(v["time"])
    v = v.groupby(["bus_id", "time"], as_index=False, sort=False, observed=True)["P_BESS"].mean()
    if vol_bess_discharge_positive:
        v["P_BESS"] = -v["P_BESS"]
    v["month"] = v["time"].dt.to_period("M")
    return v

def _bus_baseline_series(df_month: pd.DataFrame, pv_positive_is_generation: bool=False) -> pd.Series:
    base = (df_month["P_HH_HP_EV"] - df_month["P_PV"]) if pv_positive_is_generation else (df_month["P_HH_HP_EV"] + df_month["P_PV"])
    return base.groupby(df_month["bus_id"], observed=True).max()

def _bus_post_series(df_month_with_bess: pd.DataFrame, pv_positive_is_generation: bool=False) -> pd.Series:
    base = (df_month_with_bess["P_HH_HP_EV"] - df_month_with_bess["P_PV"]) if pv_positive_is_generation else (df_month_with_bess["P_HH_HP_EV"] + df_month_with_bess["P_PV"])
    post = base + df_month_with_bess.get("P_BESS", 0.0)
    return post.groupby(df_month_with_bess["bus_id"], observed=True).max()

def _bus_rng(bus_id):
    s = str(bus_id).encode()
    seed = int.from_bytes(hashlib.blake2s(s, digest_size=4).digest(), "little")
    return np.random.default_rng(seed)

def _tiny_time_weights(T, rng, scale=1.0, shape="sin"):
    t = np.arange(T)
    if shape == "sin":
        phase = rng.uniform(0, 2*np.pi)
        w = 1.0 + 0.5*np.sin(2*np.pi*t/T + phase)
    elif shape == "noise":
        w = 1.0 + rng.normal(0, 0.2, size=T)
    else:
        w = np.ones(T)
    w = np.clip(w, 0.2, None)
    return (w / w.mean()) * scale

class _Triplet:
    """Lightweight COO triplet builder."""
    def __init__(self):
        self.r = []; self.c = []; self.v = []

    def add(self, i, j, val):
        self.r.append(i); self.c.append(j); self.v.append(val)

    def coo(self, shape):
        return sparse.coo_matrix((self.v, (self.r, self.c)), shape=shape, dtype=np.float64)

# ---------- per-bus worker ----------
def _solve_ipp_one_bus(
    *,
    bus_id,
    time_values: np.ndarray,
    demand_kW: np.ndarray,
    pv_gen_kW: np.ndarray,            # PV ≥ 0 (already normalized)
    caps,
    delta_h: float,
    solver_method: str,
    epsilon_flow_per_kWh: float,
    price_import_per_kWh: float,
    price_export_penalty_per_kWh: float,
    beta_throughput_per_kWh: float,
    prefer_monthly_cyclic: bool,
    initial_soc_frac: float,
    enable_jitter: bool,
    jitter_shape_import: str,
    jitter_shape_export: str,
):
    E, Pch, Pdis, eta_c, eta_d = _coalesce_caps(caps)
    T = int(len(demand_kW))

    if T == 0 or E <= 0.0 or (Pch <= 0.0 and Pdis <= 0.0):
        return {"bus_id": bus_id, "ok": True, "msg": "no work",
                "P_BESS": np.zeros(T, np.float32),
                "soc_times": time_values, "soc_kWh": np.zeros(T, np.float32)}

    # Layout
    n_cp = n_cg = n_d = n_gp = n_pvL = n_pvG = T
    n_s  = T + 1
    o_cp  = 0
    o_cg  = o_cp  + n_cp
    o_d   = o_cg  + n_cg
    o_s   = o_d   + n_d
    o_gp  = o_s   + n_s
    o_pvL = o_gp  + n_gp
    o_pvG = o_pvL + n_pvL
    o_M   = o_pvG + n_pvG
    n_vars = o_M + 1
    sl = lambda o, L: slice(o, o+L)

    # Objective: min M + prices*Δt (+ throughput penalty if set)
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[o_M] = 1.0
    if epsilon_flow_per_kWh > 0:
        c_obj[sl(o_gp,  n_gp)]  += epsilon_flow_per_kWh * delta_h
        c_obj[sl(o_pvG, n_pvG)] += epsilon_flow_per_kWh * delta_h
    if price_import_per_kWh > 0:
        c_obj[sl(o_gp,  n_gp)]  += price_import_per_kWh * delta_h
    if price_export_penalty_per_kWh > 0:
        c_obj[sl(o_pvG, n_pvG)] += price_export_penalty_per_kWh * delta_h
    if beta_throughput_per_kWh > 0:
        c_obj[sl(o_cp, n_cp)]   += beta_throughput_per_kWh * delta_h
        c_obj[sl(o_cg, n_cg)]   += beta_throughput_per_kWh * delta_h
        c_obj[sl(o_d,  n_d )]   += beta_throughput_per_kWh * delta_h

    # Jitter (optional)
    if enable_jitter:
        rng = _bus_rng(bus_id)
        w_gp  = _tiny_time_weights(T, rng, scale=1.0, shape=jitter_shape_import)
        w_pvG = _tiny_time_weights(T, rng, scale=1.0, shape=jitter_shape_export)
        tiny_eps = 1e-6
        c_obj[sl(o_gp,  n_gp)]  += tiny_eps * w_gp  * delta_h
        c_obj[sl(o_pvG, n_pvG)] += tiny_eps * w_pvG * delta_h

    # Shapes and data
    I_T = sparse.eye(T, format="csr")
    Z   = lambda m,n: sparse.csr_matrix((m,n), dtype=np.float64)
    D   = demand_kW.astype(np.float64)
    PV  = pv_gen_kW.astype(np.float64)

    # (1) Load balance: gp - c_grid + d + pvL = D
    Aeq_bal = sparse.hstack([
        Z(T,T),      # c_pv
        -I_T,        # c_grid
        I_T,        # d
        Z(T,T+1),    # s
        I_T,         # gp
        I_T,         # pvL
        Z(T,T),      # pvG
        Z(T,1),      # M
    ], format="csr")
    beq_bal = D

    # (2) PV split: pvL + c_pv + pvG = PV
    Aeq_pv = sparse.hstack([
        I_T, Z(T,T), Z(T,T), Z(T,T+1), Z(T,T), I_T, I_T, Z(T,1)
    ], format="csr")
    beq_pv = PV

    # (3) SoC dynamics via shared blocks
    Ac, Ad, Sdiff = soc_blocks(T, delta_h, eta_c, eta_d)
    Aeq_soc = sparse.hstack([Ac, Ac, Ad, Sdiff, Z(T,T), Z(T,T), Z(T,T), Z(T,1)], format="csr")
    beq_soc = np.zeros(T, dtype=np.float64)

    # (4) Terminal row via helper
    s0_val = float(initial_soc_frac * E)
    row_s, beq_term, s0_bounds = terminal_row(T, prefer_monthly_cyclic, s0_val)
    Aeq_term = sparse.hstack([Z(1,T), Z(1,T), Z(1,T), row_s, Z(1,T), Z(1,T), Z(1,T), Z(1,1)], format="csr")

    # Stack equalities
    A_eq = sparse.vstack([Aeq_bal, Aeq_pv, Aeq_soc, Aeq_term], format="csr")
    b_eq = np.concatenate([beq_bal, beq_pv, beq_soc, beq_term]).astype(np.float64)

    # Inequalities: (A) gp_t ≤ M, (B) c_pv + c_grid ≤ Pch
    Aub_peak = sparse.hstack([ Z(T,T), Z(T,T), Z(T,T), Z(T,T+1), I_T, Z(T,T), Z(T,T), -sparse.csr_matrix(np.ones((T,1))) ], format="csr")
    bub_peak = np.zeros(T, dtype=np.float64)
    Aub_chcap = sparse.hstack([ I_T, I_T, Z(T,T), Z(T,T+1), Z(T,T), Z(T,T), Z(T,T), Z(T,1) ], format="csr")
    bub_chcap = np.full(T, Pch, dtype=np.float64)

    A_ub = sparse.vstack([Aub_peak, Aub_chcap], format="csr")
    b_ub = np.concatenate([bub_peak, bub_chcap])

    # Bounds
    bounds = []
    bounds.extend((0.0, Pch)  for _ in range(n_cp))   # c_pv
    bounds.extend((0.0, Pch)  for _ in range(n_cg))   # c_grid
    bounds.extend((0.0, Pdis) for _ in range(n_d))    # d
    bounds.append(s0_bounds)                          # s[0]
    bounds.extend((0.0, E) for _ in range(T))         # s[1..T]
    bounds.extend((0.0, None) for _ in range(n_gp))   # gp
    bounds.extend((0.0, None) for _ in range(n_pvL))  # pvL
    bounds.extend((0.0, None) for _ in range(n_pvG))  # pvG
    bounds.append((0.0, None))                        # M

    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=solver_method,
                  options={"presolve": True, "dual_feasibility_tolerance": 1e-9})
    if not res.success:
        return {"bus_id": bus_id, "ok": False, "msg": f"LP failed: {res.status} {res.message}"}

    x = res.x
    c_pv   = x[sl(o_cp,  n_cp)]
    c_grid = x[sl(o_cg,  n_cg)]
    d      = x[sl(o_d,   n_d )]
    s      = x[sl(o_s,   n_s )]

    return {"bus_id": bus_id, "ok": True, "msg": "ok",
            "P_BESS": (c_pv + c_grid - d).astype(np.float32),
            "soc_times": time_values,
            "soc_kWh": s[:-1].astype(np.float32)}

# ---------- main parallel IPP ----------
def bess_ipp_parallel(
    bus_df_scn: pd.DataFrame,
    *,
    cap_by_bus,
    delta_h: float = 0.25,
    solver_method: str = "highs",
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    epsilon_flow_per_kWh: float = 1e-6,
    price_import_per_kWh: float = 1e-3,
    price_export_penalty_per_kWh: float = 1e-3,
    beta_throughput_per_kWh: float = 0.0,
    prefer_monthly_cyclic: bool = False,
    initial_soc_frac: float = 0.25,
    enable_jitter: bool = True,
    jitter_shape_import: str = "sin",
    jitter_shape_export: str = "noise",
    verbose: bool = False,
    log_every: int = 5,
    n_workers: int = 10,
    use_processes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel IPP across buses (windowed). Returns (out_df, soc_df).
    P_BESS is charge-positive / discharge-negative.
    """
    t0 = time.perf_counter()

    need = {"bus_id", "time", "P_HH_HP_EV", "P_PV"}
    if missing := (need - set(bus_df_scn.columns)):
        raise ValueError(f"bus_df_scn missing columns: {missing}")

    df = bus_df_scn.copy()
    df["time"] = pd.to_datetime(df["time"])
    mask = _select_time_window(df, "time", time_window)
    sub = df.loc[mask].copy()

    # empty window → empty result (not full df)
    if sub.empty:
        if verbose:
            print("[IPP|PAR] Empty window – nothing to solve.")
        return df.loc[[]].assign(P_BESS=0.0), pd.DataFrame(columns=["bus_id","time","SoC_kWh"])

    # output frame only for the window
    out = sub.copy()
    out["P_BESS"] = 0.0
    soc_records = []

    groups = list(sub.groupby("bus_id", sort=True, observed=True))
    n_buses = len(groups)
    if verbose:
        w0, w1 = (df["time"].min(), df["time"].max()) if time_window is None else time_window
        print(f"[IPP|PAR] {n_buses} buses over {w0} → {w1}")

    n_workers = n_workers or max(1, os.cpu_count() or 1)
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    futures = []
    skipped = 0
    with Executor(max_workers=n_workers) as ex:
        for i, (bus_id, g) in enumerate(groups, start=1):
            idx   = g.index.to_numpy()
            t_vals= g["time"].to_numpy()
            D     = g["P_HH_HP_EV"].to_numpy(float)
            PV = ensure_pv_positive(g["P_PV"].to_numpy(float), pv_positive_is_generation=False)            
            caps  = cap_by_bus.get(bus_id, {}) if isinstance(cap_by_bus, (pd.Series, dict)) else {}

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
                    print(f"[IPP|PAR] bus {bus_id}: no BESS → skipped worker")
                continue

            fut = ex.submit(
                _solve_ipp_one_bus,
                bus_id=bus_id,
                time_values=t_vals,
                demand_kW=D,
                pv_gen_kW=PV,
                caps=caps,
                delta_h=delta_h,
                solver_method=solver_method,
                epsilon_flow_per_kWh=epsilon_flow_per_kWh,
                price_import_per_kWh=price_import_per_kWh,
                price_export_penalty_per_kWh=price_export_penalty_per_kWh,
                beta_throughput_per_kWh=beta_throughput_per_kWh,
                prefer_monthly_cyclic=prefer_monthly_cyclic,
                initial_soc_frac=initial_soc_frac,
                enable_jitter=enable_jitter,
                jitter_shape_import=jitter_shape_import,
                jitter_shape_export=jitter_shape_export,
            )
            futures.append((bus_id, idx, fut))

        for j, (bus_id, idx, fut) in enumerate(futures, start=1):
            res = fut.result()
            if not res.get("ok", False):
                raise RuntimeError(f"[IPP] LP failed for bus {bus_id}: {res.get('msg')}")
            out.loc[idx, "P_BESS"] = res["P_BESS"]
            soc_records.append(pd.DataFrame({
                "bus_id": bus_id,
                "time": pd.to_datetime(res["soc_times"]),
                "SoC_kWh": res["soc_kWh"],
            }))
            if verbose and (j == 1 or j % log_every == 0):
                print(f"[IPP|PAR] finished {j}/{len(futures)} workers")

    soc_df = pd.concat(soc_records, ignore_index=True) if soc_records else pd.DataFrame(columns=["bus_id","time","SoC_kWh"])

    if verbose:
        print(f"[IPP|PAR] Done in {time.perf_counter() - t0:.2f}s | workers={n_workers} | "
              f"backend={'proc' if use_processes else 'threads'} | skipped={skipped}")

    return out, soc_df

# ---------- monthly, per-bus gating that preserves parallelism ----------
def bess_ipp_timeseries(
    *,
    bus_df_scn: pd.DataFrame,
    vol_bus_df: pd.DataFrame,                 # VOL schedules; expected charge-positive (discharge-negative)
    vol_soc_df: pd.DataFrame | None = None,
    cap_by_bus,
    time_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    # IPP knobs (passed through):
    delta_h: float = 0.25,
    solver_method: str = "highs",
    epsilon_flow_per_kWh: float = 1e-6,
    price_import_per_kWh: float = 1e-3,
    price_export_penalty_per_kWh: float = 1e-3,
    beta_throughput_per_kWh: float = 0.0,
    prefer_monthly_cyclic: bool = False,
    initial_soc_frac: float = 0.0,
    enable_jitter: bool = True,
    jitter_shape_import: str = "sin",
    jitter_shape_export: str = "noise",
    pv_positive_is_generation: bool = False,   # PV < 0 => generation
    vol_is_discharge_positive: bool = False,   # True if VOL is discharge-positive and must be flipped
    verbose: bool = False,
):
    """
    Month-by-month gating with performance optimizations.

    Logic per bus:
      - Month 1: run IPP for all BESS buses; set global_peak_so_far[bus] = that month's post-peak.
      - Months 2..N: compute VOL post-peak; if VOL_post <= global_peak_so_far -> accept VOL;
                     else run IPP for that bus; update global_peak_so_far = max(global_peak_so_far, post_peak_this_month).
      - Non-BESS buses always accept VOL (no IPP).

    Perf tweaks:
      - Precompute VOL month frames and per-bus post-peaks once.
      - Use categoricals for bus_id/month for faster groupby/merge.
      - Deduplicate sources before joins.
    """

    # -------- Normalize & window --------
    need_cols = {"bus_id", "time", "P_HH_HP_EV", "P_PV"}
    if missing := (need_cols - set(bus_df_scn.columns)):
        raise ValueError(f"bus_df_scn missing columns: {missing}")

    df = bus_df_scn.copy()
    df["time"] = pd.to_datetime(df["time"])

    def _mask(d):
        if time_window is None:
            return np.ones(len(d), bool)
        s, e = time_window
        t = pd.to_datetime(d["time"])
        return (t >= s) & (t <= e)

    df = df.loc[_mask(df)].copy()
    if df.empty:
        if verbose:
            print("[IPP|BUS] Empty window.")
        return (
            df.assign(P_BESS=0.0),
            pd.DataFrame({"bus_id": pd.Series(dtype=object),
                          "time": pd.Series(dtype="datetime64[ns]"),
                          "SoC_kWh": pd.Series(dtype="float32")}),
            [],
        )

    # Clean & window VOL (unique (bus_id,time), normalized sign to charge-positive)
    vol = _clean_vol_schedule(vol_bus_df, vol_bess_discharge_positive=vol_is_discharge_positive)
    vol = vol.loc[_mask(vol)].copy()

    # Categoricals for speed
    df["bus_id"] = df["bus_id"].astype("category")
    vol["bus_id"] = vol["bus_id"].astype(df["bus_id"].dtype)
    df["month"] = _month_key_ser(df["time"]).astype("category")
    vol["month"] = _month_key_ser(vol["time"]).astype(df["month"].dtype)

    # Identify buses w/ BESS (E_kWh > 0)
    bus_ids = df["bus_id"].cat.categories.tolist()
    bus_ids_bess = [bid for bid in bus_ids if _coalesce_caps(cap_by_bus.get(bid, {}))[0] > 0.0]

    # Early exit: no BESS at all → return VOL passthrough
    if not bus_ids_bess:
        if verbose:
            print("[IPP|BUS] No BESS detected — returning VOL result unchanged.")
        vol_use = (
            vol[["bus_id", "time", "P_BESS"]].drop_duplicates(["bus_id", "time"], keep="last")
            if "P_BESS" in vol.columns
            else vol.assign(P_BESS=0.0)[["bus_id", "time", "P_BESS"]]
        )
        bus_out = (
            df.merge(vol_use, on=["bus_id", "time"], how="left")
              .assign(P_BESS=lambda g: g["P_BESS"].fillna(0.0))
              .sort_values(["bus_id", "time"])
        )
        # SoC passthrough (windowed) if available
        if vol_soc_df is not None and not vol_soc_df.empty:
            vsoc = vol_soc_df.copy(); vsoc["time"] = pd.to_datetime(vsoc["time"])
            soc_out = (vsoc.loc[_mask(vsoc), ["bus_id", "time", "SoC_kWh"]]
                          .sort_values(["bus_id", "time"]))
        else:
            soc_out = pd.DataFrame({"bus_id": pd.Series(dtype=object),
                                    "time": pd.Series(dtype="datetime64[ns]"),
                                    "SoC_kWh": pd.Series(dtype="float32")})
        return bus_out, soc_out, []

    # -------- Month setup --------
    months = df["month"].cat.categories.tolist()
    if verbose:
        print(f"[IPP|BUS] Months: {len(months)} | Buses: {len(bus_ids)} | BESS buses: {len(bus_ids_bess)}")

    # -------- Precompute per-month VOL frames + per-bus post-peaks (vectorized) --------
    # For each month: base gm (scenario), merge VOL once to month_frame_vol,
    # compute VOL post-peaks Series; cache both to avoid recomputation in the loop.
    month_cache = {}  # per -> dict(frame_vol, M_post_vol, gm, start, end)
    for per in months:
        gm = df.loc[df["month"] == per, ["bus_id", "time", "P_HH_HP_EV", "P_PV"]].copy()
        start_m, end_m = gm["time"].min(), gm["time"].max()
        vol_m = vol.loc[vol["month"] == per, ["bus_id", "time", "P_BESS"]].drop_duplicates(["bus_id", "time"], keep="last")
        frame_vol = gm.merge(vol_m, on=["bus_id", "time"], how="left")
        frame_vol["P_BESS"] = frame_vol["P_BESS"].fillna(0.0)
        # VOL post-peak per bus (vectorized)
        M_post_vol = _bus_post_series(frame_vol, pv_positive_is_generation=pv_positive_is_generation)
        month_cache[per] = {
            "gm": gm,
            "frame_vol": frame_vol,
            "M_post_vol": M_post_vol,
            "start": start_m,
            "end": end_m,
        }

    # -------- Iterate months implementing the gate --------
    stitched_month_frames: list[pd.DataFrame] = []
    stitched_soc_frames:   list[pd.DataFrame] = []
    bus_month_log:         list[dict] = []

    # per-bus running cap = last (max) post-peak seen so far
    global_peak_so_far: dict[Any, float] = {bid: -np.inf for bid in bus_ids_bess}
    TOL = 1e-9

    for mi, per in enumerate(months, start=1):
        cache = month_cache[per]
        gm, month_frame_vol, M_post_vol, start_m, end_m = (
            cache["gm"], cache["frame_vol"], cache["M_post_vol"], cache["start"], cache["end"]
        )

        if mi == 1:
            # Month 1: run IPP for all BESS buses
            need_buses = list(bus_ids_bess)
            if verbose:
                print(f"[IPP|BUS] {per}: month 1 → IPP for all BESS ({len(need_buses)})")
        else:
            # Months 2..N: gate on VOL post-peak vs running cap
            need_buses = []
            for bid in bus_ids_bess:
                post_vol = float(M_post_vol.get(bid, 0.0))
                if not (post_vol <= global_peak_so_far.get(bid, -np.inf) + TOL):
                    need_buses.append(bid)
            if verbose:
                print(f"[IPP|BUS] {per}: need_ipp={len(need_buses)} (BESS only) | accept_vol={len(bus_ids_bess) - len(need_buses)}")

        final_month_frame = month_frame_vol.copy()
        ipp_soc = pd.DataFrame({"bus_id": pd.Series(dtype=object),
                                "time": pd.Series(dtype="datetime64[ns]"),
                                "SoC_kWh": pd.Series(dtype="float32")})

        if need_buses:
            sub_df = gm[gm["bus_id"].isin(need_buses)].copy()
            ipp_out, ipp_soc = bess_ipp_parallel(
                sub_df,
                cap_by_bus={bid: cap_by_bus.get(bid, {}) for bid in need_buses} if isinstance(cap_by_bus, dict) else cap_by_bus,
                delta_h=delta_h,
                solver_method=solver_method,
                time_window=(start_m, end_m),
                epsilon_flow_per_kWh=epsilon_flow_per_kWh,
                price_import_per_kWh=price_import_per_kWh,
                price_export_penalty_per_kWh=price_export_penalty_per_kWh,
                beta_throughput_per_kWh=beta_throughput_per_kWh,
                prefer_monthly_cyclic=prefer_monthly_cyclic,
                initial_soc_frac=initial_soc_frac,
                enable_jitter=enable_jitter,
                jitter_shape_import=jitter_shape_import,
                jitter_shape_export=jitter_shape_export,
                verbose=False,
            )
            ipp_out["time"] = pd.to_datetime(ipp_out["time"])

            # Merge IPP first (overwrite for need_buses), VOL fallback for the rest
            final_month_frame = (
                final_month_frame
                .drop(columns=["P_BESS"])
                .merge(
                    ipp_out[["bus_id", "time", "P_BESS"]].drop_duplicates(["bus_id", "time"], keep="last"),
                    on=["bus_id", "time"], how="left"
                )
                .merge(
                    month_frame_vol[["bus_id", "time", "P_BESS"]]
                        .rename(columns={"P_BESS": "P_BESS_VOL"})
                        .drop_duplicates(["bus_id", "time"], keep="last"),
                    on=["bus_id", "time"], how="left"
                )
            )
            final_month_frame["P_BESS"] = final_month_frame["P_BESS"].fillna(final_month_frame["P_BESS_VOL"])
            final_month_frame = final_month_frame.drop(columns=["P_BESS_VOL"])

            if not ipp_soc.empty:
                # keep only windowed SoC; parallel worker already returned window, so just append
                stitched_soc_frames.append(ipp_soc.copy())

        # Update running caps from the final (IPP+VOL) frame (vectorized)
        M_post_final = _bus_post_series(final_month_frame, pv_positive_is_generation=pv_positive_is_generation)
        for bid in bus_ids_bess:
            post = float(M_post_final.get(bid, 0.0))
            prev = global_peak_so_far.get(bid, -np.inf)
            global_peak_so_far[bid] = post if post > prev else prev
            bus_month_log.append({
                "bus_id": bid,
                "month": str(per),
                "used": ("IPP" if bid in need_buses else "VOL"),
                "M_post": post,
                "cap_after": global_peak_so_far[bid],
            })

        stitched_month_frames.append(final_month_frame)

    # -------- finalize --------
    out_all = (
        pd.concat(stitched_month_frames, ignore_index=True)
          .sort_values(["bus_id", "time"])
          .drop(columns=["month"], errors="ignore")
    )
    soc_all = (
        pd.concat(stitched_soc_frames, ignore_index=True).sort_values(["bus_id", "time"])
        if stitched_soc_frames else
        pd.DataFrame({"bus_id": pd.Series(dtype=object),
                      "time": pd.Series(dtype="datetime64[ns]"),
                      "SoC_kWh": pd.Series(dtype="float32")})
    )

    return out_all, soc_all, bus_month_log