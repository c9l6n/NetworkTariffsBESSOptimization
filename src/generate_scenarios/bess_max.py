from __future__ import annotations
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linprog
import time

# -------------------------------------------------------------
# Helpers: coalesce caps, PV sign, and projection primitives
# -------------------------------------------------------------
ROUNDTRIP_EFF_DEFAULT = 0.89
MAX_CHARGE_KW_DEFAULT = 11.5
MAX_DISCHARGE_KW_DEFAULT = 11.5

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

# If your codebase already has this, import it instead of redefining
def ensure_pv_positive(pv_array: np.ndarray, pv_positive_is_generation: bool = False):
    pv = np.asarray(pv_array, dtype=float)
    return pv if pv_positive_is_generation else np.clip(-pv, 0.0, None)  # assume PV given negative for gen

def _project_sum_with_box_weighted(g, lo, hi, target, w=None, tol=1e-9, maxit=60):
    """Weighted Euclidean projection onto {p: sum p = target, lo <= p <= hi}.
       Minimizes 0.5*sum_i w_i (p_i - g_i)^2.  w_i > 0; default w_i=1."""
    g = np.asarray(g, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    if w is None: w = np.ones_like(g)
    else: w = np.asarray(w, float); w = np.maximum(w, 1e-12)

    s_lo = float(lo.sum()); s_hi = float(hi.sum())
    if target <= s_lo + 1e-12: return lo.copy()
    if target >= s_hi - 1e-12: return hi.copy()

    # s(λ) = sum clip(g - λ/w, lo, hi)  is strictly decreasing in λ
    lam_lo = np.min(w * (g - hi))
    lam_hi = np.max(w * (g - lo))
    for _ in range(maxit):
        lam = 0.5*(lam_lo + lam_hi)
        p = np.clip(g - lam / w, lo, hi)
        s = float(p.sum())
        if abs(s - target) <= tol: return p
        if s > target: lam_lo = lam   # need to reduce sum -> increase λ
        else:          lam_hi = lam   # need to increase sum -> decrease λ
    return np.clip(g - lam / w, lo, hi)

def _feasible_p_interval_from_deltas(delta_s, dt, Pch, Pdis, eta_c, eta_d):
    """
    Return [p_min, p_max] s.t. there exist c,d within power caps that realize
    the desired SoC change delta_s over dt, with efficiencies.
    Also returns a mid-interval guide.
    """
    # Typical case: eta_c*eta_d < 1
    if delta_s >= 0.0:
        # Charging net energy; parameterize by c
        c_low = delta_s / max(eta_c*dt, 1e-12)
        c_high_by_dcap = (Pdis/eta_d + (delta_s/dt)) / max(eta_c, 1e-12)
        c_low = max(0.0, c_low); c_high = min(Pch, c_high_by_dcap)
        if c_low - c_high > 1e-12: return None
        p_min = c_low  * (1.0 - eta_c*eta_d) + eta_d * (delta_s/dt)
        p_max = c_high * (1.0 - eta_c*eta_d) + eta_d * (delta_s/dt)
        return p_min, p_max, 0.5*(p_min+p_max)
    else:
        # Discharging net energy; parameterize by d
        e = -delta_s
        d_low = eta_d * e / max(dt, 1e-12)
        d_high_by_ccap = eta_d * (eta_c*Pch + e/dt)
        d_low = max(0.0, d_low); d_high = min(Pdis, d_high_by_ccap)
        if d_low - d_high > 1e-12: return None
        slope = (1.0/(eta_c*eta_d)) - 1.0
        p_low  = d_low  * slope - e/(eta_c*dt)
        p_high = d_high * slope - e/(eta_c*dt)
        p_min, p_max = (min(p_low,p_high), max(p_low,p_high))
        return p_min, p_max, 0.5*(p_min+p_max)

# -------------------------------------------------------------
# Fleet MAX (whole horizon) + Equalizing disaggregation
# -------------------------------------------------------------
def bess_max_fleet_equalized_timeseries(
    *,
    bus_df_scn: pd.DataFrame,              # columns: bus_id, time, P_HH_HP_EV, P_PV
    cap_by_bus,                            # dict or Series per bus with E_kWh, P_ch_max_kW, P_dis_max_kW, eta_c, eta_d
    time_window=None,
    delta_h: float = 0.25,                 # 15 min
    conservative_eta: bool = False,        # fleet eta = min over devices (upper bound remains valid)
    avg_over: str = "all",                 # "all" buses (recommended) or "bess" buses
    weight_mode: str = "directional",      # "directional" or "uniform"
    beta_weight: float = 2.0,              # strength for directional weighting
    pv_positive_is_generation: bool = False,
    solver_method: str = "highs-ds",       # dual-simplex tends to be fastest here
    verbose: bool = False,
):
    t0 = time.perf_counter()
    need = {"bus_id", "time", "P_HH_HP_EV", "P_PV"}
    if missing := (need - set(bus_df_scn.columns)):
        raise ValueError(f"bus_df_scn missing columns: {missing}")

    df = bus_df_scn.copy()
    df["time"] = pd.to_datetime(df["time"])
    if time_window is not None:
        s, e = time_window
        df = df[(df["time"] >= pd.to_datetime(s)) & (df["time"] <= pd.to_datetime(e))].copy()
    if df.empty:
        if verbose:
            print("[MAX-fleet] Empty window.")
        return (df.assign(P_BESS=0.0)[["bus_id","time","P_BESS"]],
                pd.DataFrame(columns=["bus_id","time","SoC_kWh"]),
                0.0,
                pd.DataFrame(columns=["time","grid_import_kW","M_block"]))

    # Baseline per bus/time
    df["PV_pos"] = ensure_pv_positive(df["P_PV"].to_numpy(float),
                                      pv_positive_is_generation=pv_positive_is_generation)
    df["D"] = df["P_HH_HP_EV"].to_numpy(float)
    df["L"] = df["D"] - df["PV_pos"]  # net baseline load per bus

    # Time grid and totals
    times = df["time"].drop_duplicates().sort_values().to_numpy("datetime64[ns]")
    T = len(times)
    if T < 2:
        raise ValueError("Not enough timesteps for MAX.")
    dt_minutes = int(round((times[1] - times[0]) / np.timedelta64(1, 'm')))
    if abs(dt_minutes - int(delta_h*60)) > 0:
        raise ValueError(f"delta_h={delta_h} does not match data sampling={dt_minutes} minutes.")

    all_bus_ids = df["bus_id"].astype("category").cat.categories.tolist()
    N_all = len(all_bus_ids)

    # Baseline totals at feeder
    base_total = (df.groupby("time", sort=True, observed=True)["L"].sum()
                    .reindex(times, fill_value=0.0).to_numpy(float))
    ub_M = float(np.maximum(base_total, 0.0).max())
    base_peak = float(np.max(np.maximum(base_total, 0.0)))

    # BESS fleet parameters
    cap_map = cap_by_bus if isinstance(cap_by_bus, (dict, pd.Series)) else {}
    E_list, Pch_list, Pdis_list, eta_c_list, eta_d_list, bess_ids = [], [], [], [], [], []
    for bid in all_bus_ids:
        E, Pch, Pdis, eta_c, eta_d = _coalesce_caps(cap_map.get(bid, {}))
        if E > 0:
            bess_ids.append(bid)
            E_list.append(E); Pch_list.append(Pch); Pdis_list.append(Pdis)
            eta_c_list.append(eta_c); eta_d_list.append(eta_d)

    if verbose:
        print(f"[MAX-fleet] Buses: {N_all} | BESS buses: {len(bess_ids)} | base_peak={base_peak:.3f} kW")

    if len(bess_ids) == 0:
        P_BESS_df = pd.DataFrame(columns=["bus_id","time","P_BESS"])
        SoC_df = pd.DataFrame(columns=["bus_id","time","SoC_kWh"])
        M_star = base_peak
        grid_df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "grid_import_kW": base_total.astype(np.float32),
            "M_block": M_star,
        })
        bus_df_max = (df[["bus_id","time","P_HH_HP_EV","P_PV"]]
                        .merge(P_BESS_df, on=["bus_id","time"], how="left")
                        .assign(P_BESS=lambda g: pd.to_numeric(g["P_BESS"], errors="coerce")
                                .fillna(0.0)
                                .astype("float32"))
                        .sort_values(["bus_id","time"])
                        .reset_index(drop=True))
        if verbose:
            print("[MAX-fleet] No BESS detected — returning baseline.")
        return bus_df_max, SoC_df, M_star, grid_df

    E_tot  = float(np.sum(E_list))
    Pch_tot  = float(np.sum(Pch_list))
    Pdis_tot = float(np.sum(Pdis_list))
    if conservative_eta:
        eta_c_fleet = float(np.min(eta_c_list))
        eta_d_fleet = float(np.min(eta_d_list))
        eta_mode = "min"
    else:
        wE = np.array(E_list, float); wE = wE / wE.sum()
        eta_c_fleet = float(np.sum(wE * np.array(eta_c_list)))
        eta_d_fleet = float(np.sum(wE * np.array(eta_d_list)))
        eta_mode = "weighted"
    if verbose:
        print(f"[MAX-fleet] Fleet caps: E={E_tot:.2f} kWh | Pch={Pch_tot:.2f} kW | Pdis={Pdis_tot:.2f} kW | "
              f"eta_c={eta_c_fleet:.3f}, eta_d={eta_d_fleet:.3f} ({eta_mode})")

    # ---------------- Fleet LP (minimize M) ----------------
    t_lp0 = time.perf_counter()
    Tn = int(T)
    o_c = 0;        o_d = o_c + Tn
    o_s = o_d + Tn; o_M = o_s + (Tn + 1)
    n_vars = o_M + 1

    c_obj = np.zeros(n_vars); c_obj[o_M] = 1.0
    bounds = []
    bounds.extend([(0.0, Pch_tot)] * Tn)
    bounds.extend([(0.0, Pdis_tot)] * Tn)
    bounds.extend([(0.0, E_tot)] * (Tn + 1))
    bounds.append((0.0, ub_M))

    # SoC dynamics + cyclic
    rows, cols, data = [], [], []; beq = []
    dh = float(delta_h)
    for t in range(Tn):
        rows += [t, t, t, t]
        cols += [o_s+t+1, o_s+t, o_c+t, o_d+t]
        data += [1.0, -1.0, -eta_c_fleet*dh, dh/max(eta_d_fleet, 1e-12)]
        beq.append(0.0)
    rows += [Tn, Tn]; cols += [o_s+Tn, o_s+0]; data += [1.0, -1.0]; beq.append(0.0)
    A_eq = sparse.coo_matrix((data, (rows, cols)), shape=(Tn+1, n_vars)).tocsr()
    b_eq = np.array(beq, float)

    # Feeder constraint: base_total[t] + c_t - d_t - M <= 0
    A_ub = sparse.lil_matrix((Tn, n_vars), dtype=float)
    for t in range(Tn):
        A_ub[t, o_c+t] = 1.0; A_ub[t, o_d+t] = -1.0; A_ub[t, o_M] = -1.0
    A_ub = A_ub.tocsr()
    b_ub = -base_total.copy()

    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=solver_method,
                  options={"presolve": True, "dual_feasibility_tolerance": 1e-9})
    t_lp = time.perf_counter() - t_lp0
    if not res.success:
        raise RuntimeError(f"[MAX-fleet] LP failed: {res.status} {res.message}")
    if verbose:
        print(f"[MAX-fleet] Fleet LP solved with {solver_method} in {t_lp:.2f}s")

    x = res.x
    c = x[o_c:o_c+Tn]; d = x[o_d:o_d+Tn]
    s = x[o_s:o_s+Tn+1]; M_star = float(x[o_M])
    P_fleet = (c - d)              # length T
    grid_import = (base_total + P_fleet).astype(np.float32)

    # Conservative clamp using proportional SoC (optional safety)
    alpha_prop = (np.array(E_list, float) / max(E_tot, 1e-12))
    s_prop = alpha_prop.reshape(-1, 1) * s.reshape(1, -1)  # (B, T+1)

    # >>> ADD THESE: make everything (B, 1) so it broadcasts with (B, T)
    E_arr   = np.asarray(E_list,  float).reshape(-1, 1)   # (B,1)
    Pch_arr = np.asarray(Pch_list, float).reshape(-1, 1)  # (B,1)
    Pdis_arr= np.asarray(Pdis_list,float).reshape(-1, 1)  # (B,1)

    den_ch = (eta_c_fleet * dh) + 1e-12
    den_dis= (dh) + 1e-12

    # per-time aggregate feasible bounds (vectors of shape (T,))
    hi_agg = np.sum(
        np.minimum(Pch_arr, np.maximum(0.0, (E_arr - s_prop[:, :-1]) / den_ch)),
        axis=0
    )
    lo_agg = -np.sum(
        np.minimum(Pdis_arr, np.maximum(0.0, (s_prop[:, :-1] * eta_d_fleet) / den_dis)),
        axis=0
    )

    before_clip_peak = float(np.max(base_total + P_fleet))
    P_fleet = np.minimum(np.maximum(P_fleet, lo_agg), hi_agg)
    after_clip_peak = float(np.max(base_total + P_fleet))

    grid_df = pd.DataFrame({"time": pd.to_datetime(times),
                            "grid_import_kW": (base_total + P_fleet).astype(np.float32),
                            "M_block": float(np.max(base_total + P_fleet))})
    if verbose:
        print(f"[MAX-fleet] Peak: base={base_peak:.3f} → fleet={before_clip_peak:.3f} → clamped={after_clip_peak:.3f} kW")

    # ---------------- Equalizing disaggregation (residual-carry) ----------------
    # Baseline load per BESS bus/time (pivot)
    L_bess = (df[df["bus_id"].isin(bess_ids)]
                .pivot_table(index="bus_id", columns="time", values="L",
                             aggfunc="first", observed=True)
                .reindex(index=bess_ids, columns=times, fill_value=0.0)
                .to_numpy(float))                            # shape (B, T)

    # Baseline load totals and average
    if avg_over.lower() == "bess":
        N_avg = len(bess_ids)
        L_total_for_avg = L_bess.sum(axis=0)                # only BESS buses
    else:
        N_avg = N_all
        L_total_for_avg = base_total                        # all buses
    L_avg = (L_total_for_avg + P_fleet) / max(N_avg, 1)

    # Device parameters in same order as bess_ids
    pars = [ _coalesce_caps(cap_map.get(b, {})) for b in bess_ids ]
    E  = np.array([p[0] for p in pars], float)
    Pch = np.array([p[1] for p in pars], float)
    Pds = np.array([p[2] for p in pars], float)
    etc = np.array([p[3] for p in pars], float)
    etd = np.array([p[4] for p in pars], float)
    B = len(bess_ids)

    P_net = np.zeros((B, T), float)
    SoC   = np.zeros((B, T+1), float)
    SoC[:, 0] = (E / max(E.sum(), 1e-12)) * s[0]

    residual = 0.0
    max_abs_residual = 0.0
    clipped_steps = 0
    t_disagg0 = time.perf_counter()

    for t in range(T):
        s_now = SoC[:, t]
        hi = np.minimum(Pch, np.maximum(0.0, (E - s_now) / (etc * dh + 1e-12)))
        lo = -np.minimum(Pds, np.maximum(0.0, (s_now * etd) / (dh + 1e-12)))

        target_raw = float(P_fleet[t]) + residual
        sum_lo = float(lo.sum()); sum_hi = float(hi.sum())
        target_eff = min(max(target_raw, sum_lo), sum_hi)
        if (target_eff != target_raw):
            clipped_steps += 1

        # Equalize toward global average
        g = (L_avg[t] - L_bess[:, t])
        if weight_mode == "directional":
            dev = L_bess[:, t] - L_avg[t]
            if target_eff < 0:
                bias = np.maximum(dev, 0.0)
            elif target_eff > 0:
                bias = np.maximum(-dev, 0.0)
            else:
                bias = np.zeros_like(dev)
            scale = np.median(np.abs(dev)) or 1.0
            w = 1.0 + float(beta_weight) * (bias / scale)
        else:
            w = np.ones(B)

        p = _project_sum_with_box_weighted(g=g, lo=lo, hi=hi, target=target_eff, w=w)
        P_net[:, t] = p

        # Update SoC
        c_step = np.maximum(p, 0.0)
        d_step = np.maximum(-p, 0.0)
        SoC[:, t+1] = s_now + etc * c_step * dh - (d_step / np.maximum(etd, 1e-12)) * dh
        SoC[:, t+1] = np.clip(SoC[:, t+1], 0.0, E)

        residual = target_raw - target_eff
        if abs(residual) > max_abs_residual:
            max_abs_residual = abs(residual)

    # Optional short backward correction
    if abs(residual) > 1e-6:
        K = min(T, int(round(24.0 / dh)))
        for t in range(T-1, max(T-K-1, -1), -1):
            s_now = SoC[:, t]
            hi = np.minimum(Pch, np.maximum(0.0, (E - s_now) / (etc * dh + 1e-12)))
            lo = -np.minimum(Pds, np.maximum(0.0, (s_now * etd) / (dh + 1e-12)))
            budget = -residual
            if abs(budget) < 1e-9:
                break
            room_lo = lo - P_net[:, t]
            room_hi = hi - P_net[:, t]
            delta = _project_sum_with_box_weighted(
                g=np.zeros(B), lo=room_lo, hi=room_hi, target=budget, w=np.ones(B)
            )
            p_new = P_net[:, t] + delta
            c_step = np.maximum(p_new, 0.0)
            d_step = np.maximum(-p_new, 0.0)
            SoC[:, t+1] = SoC[:, t] + etc * c_step * dh - (d_step / np.maximum(etd, 1e-12)) * dh
            SoC[:, t+1] = np.clip(SoC[:, t+1], 0.0, E)
            P_net[:, t] = p_new
            residual += -budget
            if abs(residual) < 1e-6:
                break

    t_disagg = time.perf_counter() - t_disagg0

    # SoC cyclicity diagnostics (end ≈ start)
    soc_start = SoC[:, 0]
    soc_end   = SoC[:, -1]
    soc_cyclic_err = np.abs(soc_end - soc_start)
    max_soc_err = float(np.max(soc_cyclic_err)) if soc_cyclic_err.size else 0.0
    mean_soc_err = float(np.mean(soc_cyclic_err)) if soc_cyclic_err.size else 0.0

    if verbose:
        clipped_pct = 100.0 * clipped_steps / max(T, 1)
        print(f"[equalize] disaggregation in {t_disagg:.2f}s | clipped_steps={clipped_steps} ({clipped_pct:.1f}%) "
              f"| max|residual|={max_abs_residual:.3f} kW | SoC cyclic err (max/mean)={max_soc_err:.3f}/{mean_soc_err:.3f} kWh")
        
    # ---- Two-pass refinement (fast) ----
    t_refine0 = time.perf_counter()

    # Compute exact aggregate bounds from realized SoC at start-of-step
    lo_sum = np.zeros(T); hi_sum = np.zeros(T)
    for t in range(T):
        s_now = SoC[:, t]
        hi = np.minimum(Pch, np.maximum(0.0, (E - s_now) / (etc * dh + 1e-12)))
        lo = -np.minimum(Pds, np.maximum(0.0, (s_now * etd) / (dh + 1e-12)))
        hi_sum[t] = float(np.sum(hi))
        lo_sum[t] = float(np.sum(lo))

    P_fleet_ref = np.minimum(np.maximum(P_fleet, lo_sum), hi_sum)

    # quick second disaggregation (no residuals)
    P_net2 = np.zeros_like(P_net)
    SoC2   = np.zeros_like(SoC)
    SoC2[:, 0] = SoC[:, 0]
    clipped2 = 0

    for t in range(T):
        s_now = SoC2[:, t]
        hi = np.minimum(Pch, np.maximum(0.0, (E - s_now) / (etc * dh + 1e-12)))
        lo = -np.minimum(Pds, np.maximum(0.0, (s_now * etd) / (dh + 1e-12)))

        target = float(P_fleet_ref[t])
        # by construction, target ∈ [sum(lo), sum(hi)]
        g = (L_avg[t] - L_bess[:, t])
        if weight_mode == "directional":
            dev = L_bess[:, t] - L_avg[t]
            bias = np.where(target < 0, np.maximum(dev, 0.0),
                            np.where(target > 0, np.maximum(-dev, 0.0), 0.0))
            scale = np.median(np.abs(dev)) or 1.0
            w = 1.0 + float(beta_weight) * (bias / scale)
        else:
            w = np.ones(B)

        p = _project_sum_with_box_weighted(g=g, lo=lo, hi=hi, target=target, w=w)
        P_net2[:, t] = p

        c_step = np.maximum(p, 0.0)
        d_step = np.maximum(-p, 0.0)
        SoC2[:, t+1] = s_now + etc * c_step * dh - (d_step / np.maximum(etd, 1e-12)) * dh
        SoC2[:, t+1] = np.clip(SoC2[:, t+1], 0.0, E)

    t_refine = time.perf_counter() - t_refine0

    # Diagnostics
    if verbose:
        # clipping and residuals
        fleet_clip_pct = 100.0 * np.mean((P_fleet != P_fleet_ref))
        soc_end_err = np.abs(SoC2[:, -1] - SoC2[:, 0])
        print(f"[refine] pass completed in {t_refine:.2f}s | "
            f"fleet power clipped {fleet_clip_pct:.1f}% of steps | "
            f"mean SoC err={np.mean(soc_end_err):.3f} kWh, max={np.max(soc_end_err):.3f} kWh")

    # replace old results
    P_net = P_net2
    SoC   = SoC2
    P_fleet = P_fleet_ref

    # Pack outputs
    times_pd = pd.to_datetime(times)
    P_rows, S_rows = [], []
    for i, bid in enumerate(bess_ids):
        P_rows.append(pd.DataFrame({"bus_id": bid, "time": times_pd, "P_BESS": P_net[i].astype(np.float32)}))
        S_rows.append(pd.DataFrame({"bus_id": bid, "time": times_pd, "SoC_kWh": SoC[i, 1:].astype(np.float32)}))
    P_BESS_df = pd.concat(P_rows, ignore_index=True)
    SoC_df    = pd.concat(S_rows,  ignore_index=True)

    # Merge with base to get full per-bus series (non-BESS have P_BESS=0)
    bus_df_max = (df[["bus_id","time","P_HH_HP_EV","P_PV"]]
                  .merge(P_BESS_df, on=["bus_id","time"], how="left")
                  .assign(P_BESS=lambda g: pd.to_numeric(g["P_BESS"], errors="coerce")
                          .fillna(0.0)
                          .astype("float32"))
                  .sort_values(["bus_id","time"])
                  .reset_index(drop=True))

    M_annual = float(np.max(grid_df["grid_import_kW"])) if "grid_df" in locals() else float(np.max(base_total + P_fleet))

    if verbose:
        total_dt = time.perf_counter() - t0
        print(f"[MAX-fleet] Done in {total_dt:.2f}s | peak base={base_peak:.2f} → optimized={M_annual:.2f} kW")

    return bus_df_max, SoC_df.sort_values(["bus_id","time"]).reset_index(drop=True), M_annual, grid_df