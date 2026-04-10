from __future__ import annotations
import os, time, hashlib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy import sparse
from scipy.optimize import linprog

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
    if caps is None: return 0.0, 0.0, 0.0, 1.0, 1.0
    if isinstance(caps, (int,float,np.floating)) or np.isscalar(caps):
        E = float(caps)
        if not np.isfinite(E) or E<=0: return 0.0,0.0,0.0,1.0,1.0
        eta = float(np.sqrt(ROUNDTRIP_EFF_DEFAULT))
        return E, MAX_CHARGE_KW_DEFAULT, MAX_DISCHARGE_KW_DEFAULT, eta, eta
    if isinstance(caps, dict):
        E = float(caps.get("E_kWh", caps.get("E", 0.0)))
        if E<=0: return 0.0,0.0,0.0,1.0,1.0
        Pch  = float(caps.get("P_ch_max_kW", caps.get("P_max_kW", MAX_CHARGE_KW_DEFAULT)))
        Pdis = float(caps.get("P_dis_max_kW", caps.get("P_max_kW", MAX_DISCHARGE_KW_DEFAULT)))
        eta_rt = float(caps.get("eta_rt", ROUNDTRIP_EFF_DEFAULT))
        eta_c = float(caps.get("eta_c", np.sqrt(eta_rt)))
        eta_d = float(caps.get("eta_d", np.sqrt(eta_rt)))
        return E, Pch, Pdis, eta_c, eta_d
    return 0.0,0.0,0.0,1.0,1.0

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
    """Minimal COO triplet builder for sparse constraints."""
    def __init__(self):
        self.r=[]; self.c=[]; self.v=[]
    def add(self, i, j, val):
        self.r.append(i); self.c.append(j); self.v.append(val)
    def coo(self, shape):
        return sparse.coo_matrix((self.v, (self.r, self.c)), shape=shape, dtype=np.float64)

def _compute_cpp_events(
    bus_df: pd.DataFrame,
    *, target_events: int,
    halo: pd.Timedelta = pd.Timedelta(minutes=30),
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    load_col: str = "P_HH_HP_EV",
    pv_col: str = "P_PV",
    bess_col: str = "P_BESS",
) -> pd.DataFrame:
    """
    Pick top system-import anchors, merge ±halo per day until target_events windows.
    """
    if bus_df.empty or target_events <= 0:
        return pd.DataFrame(columns=["event_day","event_index","start","end"])

    df = bus_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    if time_window is not None:
        s, e = time_window
        df = df[(df["time"]>=s) & (df["time"]<=e)]
        if df.empty:
            return pd.DataFrame(columns=["event_day","event_index","start","end"])

    df["net_kW"] = (df[load_col].astype(float)
                    + df[pv_col].astype(float)
                    + (df[bess_col] if bess_col in df else 0.0)).astype(float)
    df["net_import_kW"] = df["net_kW"].clip(lower=0)

    sys = (df.groupby("time", sort=True)["net_import_kW"].sum()
             .rename("sys_import_kW").reset_index())
    if sys.empty:
        return pd.DataFrame(columns=["event_day","event_index","start","end"])

    ranked = sys.sort_values(["sys_import_kW","time"], ascending=[False, True]).reset_index(drop=True)

    def merge_day_windows(halos_by_day):
        merged_by_day, total = {}, 0
        for day, spans in halos_by_day.items():
            spans = sorted(spans, key=lambda x: x[0])  # (start,end,slot_time,kw)
            merged, cur, slots, kws = [], None, [], []
            for s0,e0,t0,kw0 in spans:
                if cur is None:
                    cur=[s0,e0]; slots=[t0]; kws=[kw0]
                elif s0 <= cur[1]:
                    cur[1] = max(cur[1], e0); slots.append(t0); kws.append(kw0)
                else:
                    merged.append((pd.Timestamp(cur[0]), pd.Timestamp(cur[1]), slots[:], kws[:]))
                    cur=[s0,e0]; slots=[t0]; kws=[kw0]
            if cur is not None:
                merged.append((pd.Timestamp(cur[0]), pd.Timestamp(cur[1]), slots[:], kws[:]))
            merged_by_day[day] = merged
            total += len(merged)
        return merged_by_day, total

    halos_by_day, count = {}, 0
    for _, r in ranked.iterrows():
        if count == target_events: break
        t = pd.Timestamp(r["time"]); kw = float(r["sys_import_kW"])
        day = t.normalize(); s = t - halo; e = t + halo

        trial = {k: v[:] for k, v in halos_by_day.items()}
        trial.setdefault(day, [])
        trial[day] = trial.get(day, []) + [(s, e, t, kw)]
        _, cnt = merge_day_windows(trial)
        if cnt <= target_events:
            halos_by_day = trial; count = cnt

    merged_by_day, _ = merge_day_windows(halos_by_day)
    out = []
    for day in sorted(merged_by_day.keys()):
        for j, (s, e, slots, kws) in enumerate(merged_by_day[day], start=1):
            kmax_idx = int(np.argmax(kws))
            out.append({
                "event_day": day.date(),
                "event_index": j,
                "start": pd.Timestamp(s),
                "end": pd.Timestamp(e),
                "n_slots_merged": len(slots),
                "max_slot_time": pd.Timestamp(slots[kmax_idx]),
                "system_kw_at_max": float(kws[kmax_idx]),
            })
    return pd.DataFrame(out).sort_values("start").reset_index(drop=True)

def _event_segments(events: pd.DataFrame,
                    precharge_window: pd.Timedelta,
                    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]]=None) -> List[Dict]:
    """
    Convert event windows into merged segments with sub-windows [("pre", ...), ("event", ...)].
    """
    if events is None or events.empty: return []
    ev = events.copy()
    ev["start"] = pd.to_datetime(ev["start"]); ev["end"] = pd.to_datetime(ev["end"])
    ev = ev.sort_values("start").reset_index(drop=True)

    chunks = []
    for _, r in ev.iterrows():
        s_ev, e_ev = r["start"], r["end"]
        s_pre = s_ev - precharge_window
        if time_window is not None:
            tw_s, tw_e = time_window
            if e_ev <= tw_s or s_pre >= tw_e: continue
            s_pre = max(s_pre, tw_s); e_ev = min(e_ev, tw_e)
        if s_pre >= e_ev: continue
        win = [("pre", s_pre, s_ev), ("event", s_ev, e_ev)]
        if not chunks:
            chunks.append({"seg_start": s_pre, "seg_end": e_ev, "windows": win[:]})
            continue
        last = chunks[-1]
        if s_pre <= last["seg_end"]:
            last["seg_end"] = max(last["seg_end"], e_ev)
            last["windows"].extend(win)
        else:
            chunks.append({"seg_start": s_pre, "seg_end": e_ev, "windows": win[:]})
    for c in chunks:
        w=[]
        for kind, a, b in sorted(c["windows"], key=lambda x: (x[1], x[2], x[0])):
            if a < b: w.append((kind, a, b))
        c["windows"] = w
    return chunks

def _extract_soc_boundary(soc_vol_bus: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> Tuple[float,float]:
    s = soc_vol_bus.sort_values("time")
    times = pd.to_datetime(s["time"]).to_numpy()
    vals  = s["SoC_kWh"].to_numpy(float)
    def at(ts: pd.Timestamp) -> float:
        i = times.searchsorted(np.datetime64(ts), side="right") - 1
        return float(vals[i]) if i >= 0 else float(vals[0]) if len(vals)>0 else 0.0
    return at(t0), at(t1)

# ---------- per-bus worker ----------
def _solve_segment(
    bus_slice: pd.DataFrame,
    base_cmd: np.ndarray,
    caps,
    delta_h: float,
    pre_mask: np.ndarray,
    evt_mask: np.ndarray,
    s0: float,
    sT_target: float,
    *,
    grid_import_penalty_per_kWh: float = 1e-3,
    pv_export_penalty_per_kWh: float    = 1e-3,
    beta_throughput_per_kWh: float      = 0.0,
    jitter_import_shape: str            = "sin",
    enable_jitter: bool                 = True,
    solver_method: str                  = "highs",
    # robustness knobs
    lambda_soc: float                   = 1e-3,   # weight on |s_T - sT_target|
    allow_tiny_event_grid: bool         = True,
    tiny_event_grid_kW: float           = 1e-3,
    tiny_other_grid_kW: float           = 1e-6,   # allow ε grid charge outside pre & event
):
    """
    CPP per-segment LP (charge-positive, discharge-negative).
    Vars per t: c_pv, c_grid, d, s(t is T+1 long), gp, gm, pvL, pvG, and slack u,v for soft terminal SoC.
    Equalities:
      (i)  gp - gm - c_grid + d + pvL = D
      (ii) pvL + c_pv + pvG = PVgen
      (iii) s_{t+1} - s_t - ηc(c_pv + c_grid)Δ + (1/ηd)dΔ = 0
      (iv) s_0 = s0,  s_T - u + v = sT_target, u,v ≥ 0
    Inequalities:
      (v)  c_pv + c_grid ≤ Pch  (combined charge cap)
    Bounds:
      0 ≤ d ≤ Pdis; 0 ≤ s ≤ E; gp,gm,pvL,pvG ≥ 0.
      c_grid bounds implement "grid charging only in PRE", tiny elsewhere, tiny/zero in EVENT.
    Objective:
      minimize weighted grid import in EVENT, tiny elsewhere; nudge PV-first; small flow tie-breakers;
      soft penalty on terminal SoC mismatch via u,v.
    """
    import numpy as _np
    from scipy.optimize import linprog

    # --- unpack & guards ---
    E, Pch, Pdis, eta_c, eta_d = caps
    T = int(len(bus_slice))
    if T == 0:
        return _np.zeros(0, _np.float32), _np.zeros(1, _np.float32)

    s0 = float(_np.clip(s0, 0.0, E))
    sT_target = float(_np.clip(sT_target, 0.0, E))

    D = bus_slice["P_HH_HP_EV"].to_numpy(float)                 # load ≥ 0
    PVgen = (-bus_slice["P_PV"]).clip(lower=0).to_numpy(float)  # PV generation ≥ 0

    pre_mask = _np.asarray(pre_mask, bool)
    evt_mask = _np.asarray(evt_mask, bool)

    # --- variable layout ---
    # [ c_pv(T) | c_grid(T) | d(T) | s(T+1) | gp(T) | gm(T) | pvL(T) | pvG(T) | u(1) | v(1) ]
    n = {"c_pv":T, "c_grid":T, "d":T, "s":T+1, "gp":T, "gm":T, "pvL":T, "pvG":T, "u":1, "v":1}
    offsets = {}; off = 0
    for k in ["c_pv","c_grid","d","s","gp","gm","pvL","pvG","u","v"]:
        offsets[k] = off; off += n[k]
    n_vars = off
    def sl(k): o = offsets[k]; return slice(o, o+n[k])

    # --- objective ---
    c_obj = _np.zeros(n_vars, float)

    # Heavily weight grid imports during EVENT, tiny elsewhere
    w_outside = 1e-4
    c_obj[sl("gp")] = (evt_mask.astype(float) + w_outside * (~evt_mask).astype(float)) * delta_h

    # Nudge PV-first (discourage grid charging & PV export)
    if grid_import_penalty_per_kWh > 0:
        c_obj[sl("c_grid")] += grid_import_penalty_per_kWh * delta_h
    if pv_export_penalty_per_kWh > 0:
        c_obj[sl("pvG")]    += pv_export_penalty_per_kWh   * delta_h

    # Optional throughput penalty
    if beta_throughput_per_kWh > 0:
        c_obj[sl("c_pv")] += beta_throughput_per_kWh * delta_h
        c_obj[sl("d")]    += beta_throughput_per_kWh * delta_h

    # Tiny flow tie-breakers to aid uniqueness
    tiny = 1e-6
    c_obj[sl("gp")] += tiny * delta_h
    c_obj[sl("gm")] += tiny * delta_h

    # Deterministic jitter (per-bus) to break degeneracy nicely
    if enable_jitter:
        bus_id = int(bus_slice["bus_id"].iat[0]) if "bus_id" in bus_slice else 0
        rng = _bus_rng(bus_id)
        def _tiny_time_weights(T, rng, scale=1.0, shape="sin"):
            t = _np.arange(T)
            if shape == "sin":
                phase = rng.uniform(0, 2*_np.pi)
                w = 1.0 + 0.5*_np.sin(2*_np.pi*t/T + phase)
            elif shape == "noise":
                w = 1.0 + rng.normal(0, 0.2, size=T)
            else:
                w = _np.ones(T)
            w = _np.clip(w, 0.2, None)
            return (w / w.mean()) * scale
        w_pre  = _tiny_time_weights(T, rng, scale=1.0, shape=jitter_import_shape)
        w_else = _tiny_time_weights(T, rng, scale=1.0, shape="noise")
        c_obj[sl("gp")]     += tiny * (pre_mask * w_pre + (~evt_mask) * w_else) * delta_h
        c_obj[sl("c_grid")] += tiny * w_pre * delta_h

    # Soft terminal SoC penalty via u,v (|s_T - sT_target|)
    c_obj[offsets["u"]] += lambda_soc
    c_obj[offsets["v"]] += lambda_soc

    # --- equalities (dense row assembly; fast enough at CPP sizes) ---
    rows = []
    rhs  = []

    # (i) Power balance: gp - gm - c_grid + d + pvL = D
    for t in range(T):
        row = _np.zeros(n_vars, float)
        row[offsets["gp"]    + t] =  +1.0
        row[offsets["gm"]    + t] =  -1.0
        row[offsets["c_grid"]+ t] =  -1.0
        row[offsets["d"]     + t] =  +1.0
        row[offsets["pvL"]   + t] =  +1.0
        rows.append(row); rhs.append(D[t])

    # (ii) PV split: pvL + c_pv + pvG = PVgen
    for t in range(T):
        row = _np.zeros(n_vars, float)
        row[offsets["pvL"] + t] = 1.0
        row[offsets["c_pv"]+ t] = 1.0
        row[offsets["pvG"] + t] = 1.0
        rows.append(row); rhs.append(PVgen[t])

    # (iii) SoC dynamics: s_{t+1} - s_t - ηc(c_pv + c_grid)Δ + (1/ηd)dΔ = 0
    a = eta_c * delta_h
    b = (1.0 / eta_d) * delta_h
    for t in range(T):
        row = _np.zeros(n_vars, float)
        row[offsets["s"] + (t+1)] =  1.0
        row[offsets["s"] +  t   ] = -1.0
        row[offsets["c_pv"] + t]  = -a
        row[offsets["c_grid"]+ t] = -a
        row[offsets["d"] + t]     =  b
        rows.append(row); rhs.append(0.0)

    # (iv) Terminal constraints: s_0 = s0   and   s_T - u + v = sT_target
    row = _np.zeros(n_vars, float)
    row[offsets["s"] + 0] = 1.0
    rows.append(row); rhs.append(s0)

    row = _np.zeros(n_vars, float)
    row[offsets["s"] + T] = 1.0
    row[offsets["u"]]     = -1.0
    row[offsets["v"]]     = +1.0
    rows.append(row); rhs.append(sT_target)

    A_eq = _np.vstack(rows)
    b_eq = _np.asarray(rhs, float)

    # --- bounds ---
    bounds = []
    # c_pv: 0..Pch
    bounds.extend((0.0, Pch)  for _ in range(T))
    # c_grid: valve by mask
    for t in range(T):
        if pre_mask[t]:
            bounds.append((0.0, Pch))
        elif evt_mask[t]:
            bounds.append((0.0, tiny_event_grid_kW if allow_tiny_event_grid else 0.0))
        else:
            bounds.append((0.0, tiny_other_grid_kW))
    # d: 0..Pdis
    bounds.extend((0.0, Pdis) for _ in range(T))
    # s: 0..E (T+1)
    bounds.extend((0.0, E)    for _ in range(T+1))
    # gp, gm, pvL, pvG: ≥ 0
    bounds.extend((0.0, None) for _ in range(T))  # gp
    bounds.extend((0.0, None) for _ in range(T))  # gm
    bounds.extend((0.0, None) for _ in range(T))  # pvL
    bounds.extend((0.0, None) for _ in range(T))  # pvG
    # u, v ≥ 0
    bounds.append((0.0, None))                    # u
    bounds.append((0.0, None))                    # v

    # --- inequalities: combined charge cap c_pv + c_grid ≤ Pch ---
    if _np.isfinite(Pch):
        A_ub = _np.zeros((T, n_vars), float)
        for t in range(T):
            A_ub[t, offsets["c_pv"]  + t] = 1.0
            A_ub[t, offsets["c_grid"]+ t] = 1.0
        b_ub = _np.full(T, Pch, float)
    else:
        A_ub = None
        b_ub = None

    # --- solve (with fallback) ---
    options = {
        "presolve": True,
        "primal_feasibility_tolerance": 1e-7,
        "dual_feasibility_tolerance":   1e-7,
    }
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=solver_method, options=options)
    if not res.success:
        # fallback: IPM with slightly looser tolerances
        res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs-ipm",
                      options={"presolve": True,
                               "primal_feasibility_tolerance": 1e-6,
                               "dual_feasibility_tolerance":   1e-6})
        if not res.success:
            raise RuntimeError(f"CPP segment LP infeasible: {res.status} {res.message}")

    x = res.x
    p_bess = (x[sl("c_pv")] + x[sl("c_grid")] - x[sl("d")]).astype(_np.float32)  # charge + / discharge −
    s_all  = x[sl("s")].astype(_np.float32)                                      # length T+1 (slot-start SoC)
    return p_bess, s_all

# ---------- overlay driver ----------
def _cpp_overlay_one_bus_worker(
    *,
    bus_id,
    # per-bus time series (arrays, to keep payloads light for processes)
    g_all_times,                 # numpy array[datetime64[ns]]
    g_all_idx,                   # numpy array[int] -> index into the big bus_df for this bus
    g_all_load,                  # numpy array[float] -> P_HH_HP_EV (kW, ≥0)
    g_all_pv,                    # numpy array[float] -> P_PV (kW, NEGATIVE when generating in your pipeline)
    # SoC from VOL baseline for this bus
    soc_times,                   # numpy array[datetime64[ns]]
    soc_vals,                    # numpy array[float] -> SoC_kWh at slot start
    # tech + problem params
    caps_tuple,                  # (E, Pch, Pdis, eta_c, eta_d)
    delta_h: float,
    segments,                    # list of dicts: {seg_start, seg_end, windows=[("pre", s,e), ("event", s,e), ...]}
    grid_import_penalty_per_kWh: float  = 1e-3,
    pv_export_penalty_per_kWh: float    = 1e-3,
    beta_throughput_per_kWh: float      = 0.0,
    solver_method: str                  = "highs",
    enable_jitter: bool                 = True,
    jitter_import_shape: str            = "sin",
    enable_precharge_stagger: bool      = True,
    max_stagger_minutes: int            = 240,
):
    """
    Worker that re-optimizes this bus within each CPP segment:
      - keeps SoC boundary consistent with VOL (soft fallback if needed)
      - allows grid charging only in 'pre' windows (enforced via bounds in _solve_segment)
      - nudges PV-first via small energy penalties
    Returns:
      {
        "updates": [(seg_index_slice (np.ndarray of df indices), p_bess_seg (np.ndarray kW)), ...],
        "soc_rows": [pd.DataFrame(...), ...]   # one per segment with slot-start SoC
      }
    """
    import numpy as _np
    import pandas as _pd

    E, Pch, Pdis, eta_c, eta_d = caps_tuple

    # Helper: SoC at left boundary of any timestamp (VOL SoC is slot-start).
    # - at(ts0): SoC at the slot that starts at ts0 (left-continuous)
    # - at(ts1): SoC at the slot that starts at ts1; for a segment end we pass end+1slot outside and then
    #            take the leftmost SoC just before that (same as "SoC at end of last slot in segment").
    _soc_times = _np.asarray(soc_times)
    _soc_vals  = _np.asarray(soc_vals, float)

    def soc_at(ts: _np.datetime64) -> float:
        # index of last soc_time <= ts
        i = _np.searchsorted(_soc_times, ts, side="right") - 1
        if i < 0:
            # before first known point: clamp to first SoC
            return float(_soc_vals[0]) if len(_soc_vals) else 0.0
        if i >= len(_soc_vals):
            return float(_soc_vals[-1])
        return float(_soc_vals[i])

    # optional per-bus stagger applied to PRE windows only
    if enable_precharge_stagger and max_stagger_minutes > 0:
        rng = _bus_rng(bus_id)
        stagger = _pd.Timedelta(minutes=int(rng.integers(0, max_stagger_minutes)))
    else:
        stagger = _pd.Timedelta(0)

    times = _np.asarray(g_all_times)               # time grid for this bus
    idx_in_df = _np.asarray(g_all_idx)            # row indices in the big DataFrame
    D = _np.asarray(g_all_load, float)            # demand kW
    P_PV = _np.asarray(g_all_pv, float)           # P_PV (negative when generating)
    # PV generation as positive kW for the LP builder that expects PVgen ≥ 0
    PVgen_pos = (-P_PV).clip(min=0.0)

    updates = []
    soc_rows = []

    # quick guard
    if len(times) == 0 or len(segments) == 0:
        return {"updates": updates, "soc_rows": soc_rows}

    # common zero base_cmd (we don't actually use it in the LP; included for API compatibility)
    # If you prefer, you can keep VOL's p_bess here; functionally it's not required.
    def _base_cmd_like(n):
        return _np.zeros(n, dtype=float)

    for seg in segments:
        seg_start = _pd.Timestamp(seg["seg_start"])
        seg_end   = _pd.Timestamp(seg["seg_end"])

        # slice indexes of this bus that fall in [seg_start, seg_end)
        m = (times >= _np.datetime64(seg_start)) & (times < _np.datetime64(seg_end))
        if not m.any():
            continue

        seg_idx_positions = _np.nonzero(m)[0]        # positions within this bus arrays
        seg_df_idx        = idx_in_df[m]             # actual DataFrame indices to update
        T = int(seg_idx_positions.size)

        # local views
        seg_times = times[m]
        seg_D     = D[m]
        seg_PVpos = PVgen_pos[m]

        # build masks for "pre" and "event" windows mapped to seg_times
        pre_mask = _np.zeros(T, dtype=bool)
        evt_mask = _np.zeros(T, dtype=bool)

        # Fill masks
        for (kind, a, b) in seg["windows"]:
            a = _pd.Timestamp(a); b = _pd.Timestamp(b)
            if kind == "pre" and stagger != _pd.Timedelta(0):
                a = a - stagger  # shift earlier per-bus
            # intersection with this segment’s local time span
            s = max(a, _pd.Timestamp(seg_times[0].astype('datetime64[ns]')))
            e = min(b, _pd.Timestamp(seg_times[-1].astype('datetime64[ns]')) + _pd.Timedelta(seconds=1))
            if s >= e:
                continue
            w = (seg_times >= _np.datetime64(s)) & (seg_times < _np.datetime64(e))
            if kind == "pre":
                pre_mask |= w
            elif kind == "event":
                evt_mask |= w

        # Boundary SoC from VOL:
        s0 = soc_at(seg_times[0])
        # For terminal target, take SoC just BEFORE the first slot after seg_end
        next_after_end = _pd.Timestamp(seg_times[-1].astype('datetime64[ns]')) + _pd.Timedelta(seconds=1)
        sT = soc_at(_np.datetime64(next_after_end))

        # Build a small DataFrame for the LP routine (API expects these columns)
        seg_df = _pd.DataFrame({
            "bus_id": bus_id,
            "time": seg_times.astype('datetime64[ns]'),
            "P_HH_HP_EV": seg_D,
            "P_PV": -seg_PVpos,  # go back to your pipeline's convention (negative when generating)
        })

        base_cmd = _base_cmd_like(T)

        try:
            p_bess_seg, s_all = _solve_segment(
                seg_df, base_cmd, (E,Pch,Pdis,eta_c,eta_d), delta_h,
                pre_mask, evt_mask, s0, sT,
                grid_import_penalty_per_kWh=grid_import_penalty_per_kWh,
                pv_export_penalty_per_kWh=pv_export_penalty_per_kWh,
                beta_throughput_per_kWh=beta_throughput_per_kWh,
                jitter_import_shape=jitter_import_shape,
                enable_jitter=enable_jitter,
                solver_method=solver_method,
                lambda_soc=1e-3,
                allow_tiny_event_grid=True,
                tiny_event_grid_kW=1e-3,
                tiny_other_grid_kW=1e-6,
            )
        except RuntimeError:
            # last resort: slightly larger valves & penalty
            p_bess_seg, s_all = _solve_segment(
                seg_df, base_cmd, (E,Pch,Pdis,eta_c,eta_d), delta_h,
                pre_mask, evt_mask, s0, sT,
                grid_import_penalty_per_kWh=grid_import_penalty_per_kWh,
                pv_export_penalty_per_kWh=pv_export_penalty_per_kWh,
                beta_throughput_per_kWh=beta_throughput_per_kWh,
                jitter_import_shape=jitter_import_shape,
                enable_jitter=enable_jitter,
                solver_method="highs-ipm",
                lambda_soc=1e-2,
                allow_tiny_event_grid=True,
                tiny_event_grid_kW=1e-2,
                tiny_other_grid_kW=1e-4,
            )
        
        # record update for this contiguous slice
        updates.append((seg_df_idx, p_bess_seg.astype(_np.float32)))

        # store slot-start SoC for these timestamps (length T, since s_all is length T+1)
        soc_rows.append(_pd.DataFrame({
            "bus_id": bus_id,
            "time": seg_times.astype('datetime64[ns]'),
            "SoC_kWh": _np.asarray(s_all[:-1], dtype=_np.float32),
        }))

    return {"updates": updates, "soc_rows": soc_rows}

# ---------- main parallel CPP ----------
def bess_cpp_parallel(
    bus_df_vol: pd.DataFrame,
    soc_vol: pd.DataFrame,
    cap_by_bus: pd.Series | Dict,
    *,
    target_events: int = 10,
    halo: pd.Timedelta = pd.Timedelta(minutes=30),
    events: Optional[pd.DataFrame] = None,
    precharge_window: pd.Timedelta = pd.Timedelta(hours=24),
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    delta_h: float = 0.25,
    verbose: bool = False,
    grid_import_penalty_per_kWh: float  = 1e-3,
    pv_export_penalty_per_kWh: float    = 1e-3,
    beta_throughput_per_kWh: float      = 0.0,
    solver_method: str                  = "highs",
    enable_jitter: bool                 = True,
    jitter_import_shape: str            = "sin",
    enable_precharge_stagger: bool      = False,
    max_stagger_minutes: int            = 60,
    # parallel knobs
    n_workers: int = 10,
    use_processes: bool = False,
):
    t0 = time.perf_counter()

    df = bus_df_vol.copy()
    df["time"] = pd.to_datetime(df["time"])
    if "P_BESS" not in df.columns:
        raise ValueError("bus_df_vol must contain VOL results in 'P_BESS'")

    # compute or normalize events (driver-side, once)
    if events is None:
        events = _compute_cpp_events(
            bus_df=df, target_events=target_events, halo=halo, time_window=time_window,
            load_col="P_HH_HP_EV", pv_col="P_PV", bess_col="P_BESS"
        )
        if verbose:
            print(f"[CPP|PAR] computed {len(events)} events (target={target_events}, halo={halo}).")
    else:
        events = events.copy()
        events["start"] = pd.to_datetime(events["start"]); events["end"] = pd.to_datetime(events["end"])

    # build merged segments (driver-side)
    segs = _event_segments(events, precharge_window, time_window=time_window)
    if verbose:
        print(f"[CPP|PAR] segments={len(segs)}")

    out = df.copy()
    soc_rows_all = []

    # Prepare groups
    if time_window is not None:
        s_w, e_w = time_window
        df = df[(df["time"]>=s_w) & (df["time"]<e_w)].copy()

    groups = list(df.groupby("bus_id", sort=False))
    n_buses = len(groups)
    if verbose:
        print(f"[CPP|PAR] buses to process: {n_buses}")

    # Executor selection
    n_workers = n_workers or max(1, os.cpu_count() or 1)
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    futures = []
    skipped = 0
    with Executor(max_workers=n_workers) as ex:
        for bus_id, g_all in groups:
            E, Pch, Pdis, eta_c, eta_d = _coalesce_caps(cap_by_bus.get(bus_id, {}))
            if E <= 0.0 or (Pch <= 0.0 and Pdis <= 0.0):
                skipped += 1
                continue

            # Slice soc for this bus
            soc_bus = soc_vol.loc[soc_vol["bus_id"]==bus_id, ["time","SoC_kWh"]].copy()
            if soc_bus.empty:
                continue
            soc_bus["time"] = pd.to_datetime(soc_bus["time"])

            fut = ex.submit(
                _cpp_overlay_one_bus_worker,
                bus_id=bus_id,
                g_all_times=g_all["time"].to_numpy(),
                g_all_idx=g_all.index.to_numpy(),
                g_all_load=g_all["P_HH_HP_EV"].to_numpy(float),
                g_all_pv=g_all["P_PV"].to_numpy(float),
                soc_times=soc_bus["time"].to_numpy(),
                soc_vals=soc_bus["SoC_kWh"].to_numpy(float),
                caps_tuple=(E, Pch, Pdis, eta_c, eta_d),
                delta_h=delta_h,
                segments=segs,
                grid_import_penalty_per_kWh=grid_import_penalty_per_kWh,
                pv_export_penalty_per_kWh=pv_export_penalty_per_kWh,
                beta_throughput_per_kWh=beta_throughput_per_kWh,
                solver_method=solver_method,
                enable_jitter=enable_jitter,
                jitter_import_shape=jitter_import_shape,
                enable_precharge_stagger=enable_precharge_stagger,
                max_stagger_minutes=max_stagger_minutes,
            )
            futures.append(fut)

        # collect
        for j, fut in enumerate(futures, start=1):
            res = fut.result()
            # apply updates
            for idx_slice, p_bess in res["updates"]:
                out.loc[idx_slice, "P_BESS"] = p_bess
            # SoC slices
            soc_rows_all.extend(res["soc_rows"])
            if verbose and (j == 1 or j % 10 == 0):
                print(f"[CPP|PAR] finished {j}/{len(futures)} buses")

    # If no segments overlapped a bus, keep its VOL SoC
    soc_cpp = (pd.concat(soc_rows_all, ignore_index=True)
               if soc_rows_all else soc_vol.copy()[["bus_id","time","SoC_kWh"]])
    soc_cpp["time"] = pd.to_datetime(soc_cpp["time"])
    out["time"] = pd.to_datetime(out["time"])

    segments_df = pd.DataFrame([
        {"seg_start": s["seg_start"], "seg_end": s["seg_end"], "n_subwindows": len(s["windows"])}
        for s in segs
    ])

    if verbose:
        print(f"[CPP|PAR] Done in {time.perf_counter()-t0:.2f}s | workers={n_workers} | "
              f"backend={'proc' if use_processes else 'threads'} | skipped_no_BESS={skipped}")

    return out, soc_cpp, events, segments_df