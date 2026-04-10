import numpy as np
from scipy import sparse

def ensure_pv_positive(PV_raw: np.ndarray, pv_positive_is_generation: bool) -> np.ndarray:
    """
    Returns PV >= 0 (generation) regardless of input convention.
    """
    PV_raw = np.asarray(PV_raw, float)
    return PV_raw if pv_positive_is_generation else (-PV_raw).clip(min=0.0)

def soc_blocks(T: int, delta_h: float, eta_c: float, eta_d: float):
    """
    Returns the 3 blocks used in: s_{t+1} - s_t - eta_c*Δ*c + (1/eta_d)*Δ*d = 0
    Shapes: Ac:(T×T), Ad:(T×T), As:(T×(T+1)) in CSR.
    """
    I_T = sparse.eye(T, format="csr")
    Sdiff = sparse.diags([-np.ones(T), np.ones(T)], [0,1], shape=(T, T+1), format="csr")
    Ac = (-(eta_c * delta_h)) * I_T
    Ad = (( 1.0 / eta_d) * delta_h) * I_T
    return Ac, Ad, Sdiff

def terminal_row(T: int, prefer_cyclic: bool, s0: float):
    """
    Returns (Aeq_row_on_s, rhs) for either: s_T - s_0 = 0  (cyclic) or s_0 = s0 (fixed).
    The row has length (T+1) and is intended to be placed in the S-block columns.
    """
    if prefer_cyclic:
        row = sparse.csr_matrix(([1.0, -1.0], ([0, 0], [T, 0])), shape=(1, T+1))
        rhs = np.array([0.0], float)
        s0_bounds = (0.0, None)  # s0 free in bounds
    else:
        row = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, T+1))
        rhs = np.array([s0], float)
        s0_bounds = (s0, s0)     # fix s0 in bounds
    return row, rhs, s0_bounds

def gp_peak_rows(T: int):
    """
    Rows for gp_t - M <= 0  (T inequalities). Returns CSR of shape (T × (T + 1)),
    where the "+1" is for the 'M' column. To place it, hstack with other variable blocks.
    """
    I_T = sparse.eye(T, format="csr")
    Mcol = -sparse.csr_matrix(np.ones((T,1)))
    return sparse.hstack([I_T, Mcol], format="csr")