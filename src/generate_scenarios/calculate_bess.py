import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.generate_scenarios import bess_vol as vol
from src.generate_scenarios import bess_max as max
from src.generate_scenarios import bess_ipp as ipp
from src.generate_scenarios import bess_cpp as cpp

def attribute_pv_bess_households(
    bus_df: pd.DataFrame,
    hh_df_scn: pd.DataFrame,
    *,
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    peak_time: Optional[pd.Timestamp] = None,
    events: Optional[pd.DataFrame] = None,
    delta_t: float = 0.25,
    roundtrip_eff: float = 0.89,
    return_timeseries: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attribute PV/BESS at building (bus) level down to households over a selected window.

    Returns:
      hh_ts      : per-HH timeseries (kW) over the window (if return_timeseries=True)
      hh_window  : window summary per HH (kWh + individual_window_peak_kW)
    """
    # --- filter to window ---
    bus = bus_df.copy()
    hh  = hh_df_scn.copy()
    bus["time"] = pd.to_datetime(bus["time"])
    hh["time"]  = pd.to_datetime(hh["time"])
    if time_window is not None:
        s, e = time_window
        bus = bus[(bus["time"] >= s) & (bus["time"] <= e)]
        hh  = hh [(hh ["time"] >= s) & (hh ["time"] <= e)]

    # --- join HH rows to bus-level signals ---
    use_cols = ["bus_id","time","P_HH_HP_EV","P_PV","P_BESS"]
    df = hh.merge(bus[use_cols], on=["bus_id","time"], how="left", validate="many_to_one").copy()

    # HH own loads (kW)
    df["load_h"] = (df["P_HH"].astype(np.float32)
                    + df["P_HP"].astype(np.float32)
                    + df["P_EV"].astype(np.float32))

    # Bus-level PV generation as +kW
    df["PV_gen"] = (-df["P_PV"]).clip(lower=0).astype(np.float32)

    # Surplus PV at the bus after serving bus demand
    df["bus_PV_surplus"] = (df["PV_gen"] - df["P_HH_HP_EV"]).clip(lower=0).astype(np.float32)

    # BESS charge/discharge at the bus (+charge, -discharge)
    df["BESS_charge"]     = ( df["P_BESS"]).clip(lower=0).astype(np.float32)
    df["BESS_discharge"]  = (-df["P_BESS"]).clip(lower=0).astype(np.float32)

    # PV-driven charge first, remainder from grid
    df["BESS_charge_PV"]     = np.minimum(df["bus_PV_surplus"], df["BESS_charge"]).astype(np.float32)
    df["BESS_charge_import"] = (df["BESS_charge"] - df["BESS_charge_PV"]).clip(lower=0).astype(np.float32)

    # Simple roundtrip loss model (charge-side)
    df["BESS_loss"] = (df["BESS_charge"] * (1.0 - roundtrip_eff)).clip(lower=0).astype(np.float32)

    # Net bus import/export (kW) BEFORE allocation
    # net = load - PV + charge - discharge  (consistent with your sign convention)
    df["bus_net_import"] = (df["P_HH_HP_EV"].astype(np.float32)
                            - df["PV_gen"]
                            + df["BESS_charge"]
                            - df["BESS_discharge"]).astype(np.float32)
    df["bus_import_kW"] = df["bus_net_import"].clip(lower=0)
    df["bus_export_kW"] = (-df["bus_net_import"]).clip(lower=0)

    # Allocation ratio: HH share of bus demand (guard against zero)
    den = df["P_HH_HP_EV"].astype(np.float64)
    num = df["load_h"].astype(np.float64)
    df["ratio"] = np.divide(num, den, out=np.zeros_like(num), where=den > 0).astype(np.float32)

    # Allocate to HH (kW)
    df["pv_generation_kW"]         = (df["PV_gen"] * df["ratio"]).astype(np.float32)
    df["pv_direct_consumption_kW"] = ((df["PV_gen"] - df["bus_PV_surplus"]) * df["ratio"]).astype(np.float32)
    df["bess_charge_kW"]           = (df["BESS_charge"] * df["ratio"]).astype(np.float32)
    df["bess_charge_pv_kW"]        = (df["BESS_charge_PV"] * df["ratio"]).astype(np.float32)
    df["bess_charge_import_kW"]    = (df["BESS_charge_import"] * df["ratio"]).astype(np.float32)
    df["bess_loss_kW"]             = (df["BESS_loss"] * df["ratio"]).astype(np.float32)
    df["bess_discharge_kW"]        = (df["BESS_discharge"] * df["ratio"]).astype(np.float32)

    # Allocate bus net import/export to HH (kW)
    df["import_kW"] = (df["bus_import_kW"] * df["ratio"]).astype(np.float32)
    df["export_kW"] = (df["bus_export_kW"] * df["ratio"]).astype(np.float32)

    # CPP event imports (optional)
    if events is None or (isinstance(events, pd.DataFrame) and events.empty):
        df["import_in_cpp_events_kW"] = 0.0
    else:
        ev = events[["start","end"]].copy()
        ev["start"] = pd.to_datetime(ev["start"]); ev["end"] = pd.to_datetime(ev["end"])
        t = df["time"].to_numpy("datetime64[ns]")
        mask = np.zeros(len(df), dtype=bool)
        for s, e in zip(ev["start"].to_numpy("datetime64[ns]"), ev["end"].to_numpy("datetime64[ns]")):
            mask |= (t >= s) & (t < e)
        df["import_in_cpp_events_kW"] = np.where(mask, df["import_kW"], 0.0).astype(np.float32)

    # ---------------- window summary (kWh) ----------------
    agg = {
        "load_h"                 : lambda x: x.sum() * delta_t,
        "P_HH"                   : lambda x: x.sum() * delta_t,
        "P_HP"                   : lambda x: x.sum() * delta_t,
        "P_EV"                   : lambda x: x.sum() * delta_t,
        "pv_generation_kW"       : lambda x: x.sum() * delta_t,
        "pv_direct_consumption_kW": lambda x: x.sum() * delta_t,
        "bess_charge_kW"         : lambda x: x.sum() * delta_t,
        "bess_charge_pv_kW"      : lambda x: x.sum() * delta_t,
        "bess_charge_import_kW"  : lambda x: x.sum() * delta_t,
        "bess_loss_kW"           : lambda x: x.sum() * delta_t,
        "bess_discharge_kW"      : lambda x: x.sum() * delta_t,
        "import_kW"              : lambda x: x.sum() * delta_t,
        "export_kW"              : lambda x: x.sum() * delta_t,
        "import_in_cpp_events_kW": lambda x: x.sum() * delta_t,
    }
    hh_window = (df.groupby(["bus_id","hh_id"], as_index=False, observed=True)
                   .agg(**{k.replace("_kW","_kWh"): (k, fn) for k, fn in agg.items()},
                        individual_window_peak_kW=("import_kW", "max"),
                        coincident_window_peak_kW=("import_kW", lambda x: x[df["time"] == peak_time].max() if peak_time is not None else np.nan)))

    # Return both (timeseries optional)
    if return_timeseries:
        # keep a tidy selection of kW columns
        keep_cols = ["bus_id","hh_id","time","load_h","pv_generation_kW","pv_direct_consumption_kW",
                     "bess_charge_kW","bess_charge_pv_kW","bess_charge_import_kW","bess_loss_kW",
                     "bess_discharge_kW","import_kW","export_kW","import_in_cpp_events_kW"]
        hh_ts = df[keep_cols].sort_values(["bus_id","hh_id","time"]).reset_index(drop=True)
        return hh_ts, hh_window
    else:
        return pd.DataFrame(), hh_window


def build_top_n_peak_tables(
        bus_df: pd.DataFrame, 
        hh_df: pd.DataFrame, 
        top_n: int = 10
) -> Tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # copies & dtypes
    bus = bus_df.copy()
    hh  = hh_df.copy()

    # bus-level total net power
    bus['P_Total'] = (
        bus['P_HH_HP_EV'].astype(np.float64) +
        bus['P_PV'].astype(np.float64) +
        bus['P_BESS'].astype(np.float64)
    )

    # system peak times (sum across buses per timestamp)
    sys_peaks = (bus.groupby('time', as_index=False, observed=True)['P_Total'].sum()
               .sort_values('P_Total', ascending=False)
               .reset_index(drop=True)
               .rename(columns={'P_Total':'system_P_Total'}))
    
    # keep time as datetime, add rank for sorting downstream
    sys_peaks = sys_peaks.assign(peak_rank=np.arange(1, len(sys_peaks)+1))
    peak_times = sys_peaks['time']

    # BUS: rows at peak times (not aggregated)
    top_n_bus_df = bus[bus['time'].isin(peak_times)].copy()
    top_n_bus_df = (top_n_bus_df
        .merge(sys_peaks[['time','peak_rank']], on='time', how='left')
        .sort_values(['peak_rank','bus_id'])
        .drop(columns='peak_rank')
        .reset_index(drop=True)
    )

    # BUS: aggregated per (bus_id, time)
    top_n_bus_loads = (
        bus[bus['time'].isin(peak_times)]
        .groupby(['bus_id','time'], as_index=False, observed=True)['P_Total'].sum()
        .merge(sys_peaks[['time','peak_rank']], on='time', how='left')
        .sort_values(['peak_rank','P_Total'], ascending=[True, False])
        .drop(columns='peak_rank')
        .reset_index(drop=True)
    )

    # HH: ensure HH+HP+EV is present
    if 'P_HH_HP_EV' not in hh.columns:
        must = {'P_HH','P_HP','P_EV'}
        if not must.issubset(hh.columns):
            missing = must - set(hh.columns)
            raise ValueError(f"hh_df missing columns to compute P_HH_HP_EV: {missing}")
        hh['P_HH_HP_EV'] = hh[['P_HH','P_HP','P_EV']].sum(axis=1)

    # HH: aggregated per (bus_id, hh_id, time) at those timestamps
    top_n_hh_loads = (
        hh[hh['time'].isin(peak_times)]
        .groupby(['bus_id','hh_id','time'], as_index=False, observed=True)['P_HH_HP_EV'].sum()
        .merge(sys_peaks[['time','peak_rank']], on='time', how='left')
        .sort_values(['peak_rank','P_HH_HP_EV'], ascending=[True, False])
        .drop(columns='peak_rank')
        .reset_index(drop=True)
    )

    # final: return peak times as a DF (datetime intact) plus the three tables
    peak_times_df = sys_peaks[['time','system_P_Total']]

    return peak_times_df, top_n_bus_df, top_n_bus_loads, top_n_hh_loads

def _pack_result(
    *,
    hh_attr: pd.DataFrame,
    peak_times: pd.DataFrame,
    top_n_bus_df: pd.DataFrame,
    top_n_bus_loads: pd.DataFrame,
    top_n_hh_loads: pd.DataFrame,
    vis_output: pd.DataFrame
) -> Dict[str, Any]:
    """Standardized output container for all algorithms."""
    return {
        "HH_attribution": hh_attr,
        "peak_times": peak_times,
        "peak_times_df": top_n_bus_df,
        "peak_times_bus_loads": top_n_bus_loads,
        "peak_times_hh_loads": top_n_hh_loads,
        "vis_output": vis_output
    }

# --------------------------------------
# Main function: VOL
# --------------------------------------
def run_VOL_algorithm(
    bus_df_scn: pd.DataFrame,
    hh_df_scn: pd.DataFrame,
    cap_by_bus: pd.Series,
    time_window: Tuple[pd.Timestamp, pd.Timestamp]
) -> Dict[str, Any]:

    # 1) Dispatch (VOL baseline)
    bus_df_vol, soc_vol = vol.bess_vol_auto_window(
        bus_df_scn=bus_df_scn,           
        cap_by_bus=cap_by_bus,          
        time_window=time_window,           
        prefer_monthly_cyclic=True, 
    )

    # 2) Peak tables
    (peak_times_vol,
     top_n_bus_df_vol,
     top_n_bus_loads_vol,
     top_n_hh_loads_vol) = build_top_n_peak_tables(bus_df_vol, hh_df_scn)
    
    peak_time = peak_times_vol.loc[0].time
    
    # 3) Distribution per household (VOL)
    _, hh_attr_vol = attribute_pv_bess_households(bus_df_vol, hh_df_scn, time_window=time_window, peak_time=peak_time)
    
    # OPTIONAL: Output for visualization
    start = time_window[0]
    end   = time_window[1]

    vol_viz = (
        bus_df_vol.loc[bus_df_vol["time"].between(start, end)].copy()
        .assign(total_vol=lambda d: d["P_HH_HP_EV"] + d["P_PV"] + d["P_BESS"])
        .groupby("time", as_index=False, observed=True)["total_vol"].sum()
    )

    return _pack_result(
        hh_attr=hh_attr_vol,
        peak_times=peak_times_vol,
        top_n_bus_df=top_n_bus_df_vol,
        top_n_bus_loads=top_n_bus_loads_vol,
        top_n_hh_loads=top_n_hh_loads_vol,
        vis_output = vol_viz
    ), bus_df_vol, soc_vol

# --------------------------------------
# Main function: MAX
# --------------------------------------
def run_MAX_algorithm(
    bus_df_scn: pd.DataFrame,
    hh_df_scn: pd.DataFrame,
    cap_by_bus: pd.Series,
    time_window: Tuple[pd.Timestamp, pd.Timestamp]
) -> Dict[str, Any]:

    bus_df_max, soc_max, M_annual, log = max.bess_max_fleet_equalized_timeseries(
        bus_df_scn=bus_df_scn,       
        cap_by_bus=cap_by_bus,       
        time_window=time_window,
    )

    # 2) Peak tables
    (peak_times_max,
     top_n_bus_df_max,
     top_n_bus_loads_max,
     top_n_hh_loads_max) = build_top_n_peak_tables(bus_df_max, hh_df_scn)
    
    peak_time = peak_times_max.loc[0].time

    # 3) Distribution per household (MAX)
    _, hh_attr_max = attribute_pv_bess_households(bus_df_max, hh_df_scn, time_window=time_window, peak_time=peak_time)
    
    # OPTIONAL: Output for visualization
    start = time_window[0]
    end   = time_window[1]

    max_viz = (
        bus_df_max.loc[bus_df_max["time"].between(start, end)].copy()
        .assign(total_max=lambda d: d["P_HH_HP_EV"] + d["P_PV"] + d["P_BESS"])
        .groupby("time", as_index=False, observed=True)["total_max"].sum()
    )

    return _pack_result(
        hh_attr=hh_attr_max,
        peak_times=peak_times_max,
        top_n_bus_df=top_n_bus_df_max,
        top_n_bus_loads=top_n_bus_loads_max,
        top_n_hh_loads=top_n_hh_loads_max,
        vis_output = max_viz
    )

# --------------------------------------
# Main function: IPP
# --------------------------------------
def run_IPP_algorithm(
    bus_df_scn: pd.DataFrame,
    bus_df_vol: pd.DataFrame,
    soc_vol: pd.DataFrame,
    hh_df_scn: pd.DataFrame,
    cap_by_bus: pd.Series,
    time_window: Tuple[pd.Timestamp, pd.Timestamp]
) -> Dict[str, Any]:

    bus_df_ipp, soc_ipp, bus_month_log = ipp.bess_ipp_timeseries(
        bus_df_scn=bus_df_scn,       
        vol_bus_df=bus_df_vol,
        vol_soc_df=soc_vol,       
        cap_by_bus=cap_by_bus,
        time_window=time_window,       
    )

    # 2) Peak tables
    (peak_times_ipp,
     top_n_bus_df_ipp,
     top_n_bus_loads_ipp,
     top_n_hh_loads_ipp) = build_top_n_peak_tables(bus_df_ipp, hh_df_scn)
    
    peak_time = peak_times_ipp.loc[0].time

    # 3) Distribution per household (IPP)
    _, hh_attr_ipp = attribute_pv_bess_households(bus_df_ipp, hh_df_scn, time_window=time_window, peak_time=peak_time)

    # OPTIONAL: Output for visualization
    start = time_window[0]
    end   = time_window[1]

    ipp_viz = (
        bus_df_ipp.loc[bus_df_ipp["time"].between(start, end)].copy()
        .assign(total_ipp=lambda d: d["P_HH_HP_EV"] + d["P_PV"] + d["P_BESS"])
        .groupby("time", as_index=False, observed=True)["total_ipp"].sum()
    )

    return _pack_result(
        hh_attr=hh_attr_ipp,
        peak_times=peak_times_ipp,
        top_n_bus_df=top_n_bus_df_ipp,
        top_n_bus_loads=top_n_bus_loads_ipp,
        top_n_hh_loads=top_n_hh_loads_ipp,
        vis_output = ipp_viz
    )

# --------------------------------------
# Main function: CPP
# --------------------------------------
def run_CPP_algorithm(
    bus_df_vol: pd.DataFrame,
    soc_vol: pd.DataFrame,
    hh_df_scn: pd.DataFrame,
    cap_by_bus: pd.Series,
    time_window: Tuple[pd.Timestamp, pd.Timestamp],
    *,
    target_events: int = 10,
) -> Dict[str, Any]:

    bus_df_cpp, soc_cpp, events_cpp, segments_cpp = cpp.bess_cpp_parallel(
        bus_df_vol, soc_vol, cap_by_bus,
        target_events=target_events,
        halo=pd.Timedelta(minutes=60),
        precharge_window=pd.Timedelta(hours=24),
        time_window=time_window, 
    )

    # 2) Peak tables
    (peak_times_cpp,
     top_n_bus_df_cpp,
     top_n_bus_loads_cpp,
     top_n_hh_loads_cpp) = build_top_n_peak_tables(bus_df_cpp, hh_df_scn)

    peak_time = peak_times_cpp.loc[0].time

    # 3) Distribution per household (CPP)
    _, hh_attr_cpp = attribute_pv_bess_households(bus_df_cpp, hh_df_scn, time_window=time_window, events=events_cpp, peak_time=peak_time)

    # OPTIONAL: Output for visualization
    start = time_window[0]
    end   = time_window[1]

    cpp_viz = (
        bus_df_cpp.loc[bus_df_cpp["time"].between(start, end)].copy()
        .assign(total_cpp=lambda d: d["P_HH_HP_EV"] + d["P_PV"] + d["P_BESS"])
        .groupby("time", as_index=False, observed=True)["total_cpp"].sum()
    )

    return _pack_result(
        hh_attr=hh_attr_cpp,
        peak_times=peak_times_cpp,
        top_n_bus_df=top_n_bus_df_cpp,
        top_n_bus_loads=top_n_bus_loads_cpp,
        top_n_hh_loads=top_n_hh_loads_cpp,
        vis_output = cpp_viz
    )