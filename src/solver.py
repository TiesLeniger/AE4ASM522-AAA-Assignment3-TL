import numpy as np
from scipy.sparse.linalg import gmres
from src.vortex import *
from src.wing import TangDowellWing
from typing import Union

def make_bound_im(wing: TangDowellWing) -> np.ndarray:
    v_ind, w_ind = v_ind_by_vortex_ring(wing.panel_cntrl, wing.A, wing.B, wing.C, wing.D, gamma = 1, return_w=True)
    im = np.sum(v_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
    dwm = np.sum(w_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)

    return im, dwm

def make_wake_im(wing: TangDowellWing, alpha: float, l_wake_c: int) -> np.ndarray:
    TE_wake_points = np.concatenate((wing.A[-1, :, :], wing.D[-1, -1, :][None, :]), axis = 0)
    wake_points = TE_wake_points + l_wake_c*wing.chord*np.array([np.cos(alpha), 0.0, np.sin(alpha)])

    v_ind, w_ind = v_ind_by_vortex_ring(wing.panel_cntrl, wake_points[:-1], TE_wake_points[:-1], TE_wake_points[1:], wake_points[1:], gamma = 1, return_w=True)
    wake_im = np.sum(v_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
    wake_dwm = np.sum(w_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)

    return wake_im, wake_dwm

def make_right_hand_side_static(wing: TangDowellWing, v_inf: float, alpha: float) -> np.ndarray:
    
    v_inf = (v_inf * np.array([np.cos(alpha), 0.0, np.sin(alpha)]))[None, :]
    rhs = np.sum(-v_inf * wing.panel_normal.reshape(-1, 3), axis = 1, keepdims = False)

    return rhs

def make_right_hand_side(wing: TangDowellWing, v_inf: float, alpha: float, xi: np.ndarray, mapping_matrix: np.ndarray) -> np.ndarray:

    v_inf_norm = v_inf
    v_inf = (v_inf * np.array([np.cos(alpha), 0.0, np.sin(alpha)]))[None, :]
    rhs = np.sum(-v_inf * wing.panel_normal.reshape(-1, 3), axis = 1, keepdims = False)
    displacement_term = mapping_matrix @ xi
    displacement_term = np.concatenate((np.flip(displacement_term, axis = 0), displacement_term), axis = 0)
    rhs += v_inf_norm * displacement_term

    return rhs

def generate_solutions(wing: TangDowellWing, v_inf: float, alpha: float, l_wake_c: Union[int, np.ndarray]) -> np.ndarray:

    te_panels_idx = (wing.n_c - 1)*wing.n_s
    bound_im, bound_dw = make_bound_im(wing)
    l_wake_c = np.atleast_1d(l_wake_c)

    results = []

    for length in l_wake_c:
        wake_im, wake_dw = make_wake_im(wing, alpha, length)
        full_im, full_dw = np.copy(bound_im), np.copy(bound_dw)
        full_im[:, te_panels_idx:] += wake_im
        full_dw[:, te_panels_idx:] += wake_dw
        rhs = make_right_hand_side_static(wing, v_inf, alpha)
        gamma, info = gmres(full_im, rhs)

        if info != 0:
            raise RuntimeError(f"GMRES did not converge. Info code: {info}")
        
        solution = {'alpha': alpha, 'wake_length': length, 'gamma': gamma, 'downwash': full_dw @ gamma}
        results.append(solution)
    
    return results