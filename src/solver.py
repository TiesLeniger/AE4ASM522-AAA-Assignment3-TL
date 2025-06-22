import numpy as np
from scipy.sparse.linalg import gmres
from src.vortex import *
from src.wing import TangDowellWing
from typing import Union

def make_bound_im(wing: TangDowellWing) -> np.ndarray:
    v_ind, w_ind = v_ind_by_vortex_ring(wing.panel_ctrl, wing.A, wing.B, wing.C, wing.D, gamma = 1, return_w=True)
    half_im_matrix = np.sum(v_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
    full_im_matrix = np.concatenate((np.flip(half_im_matrix, axis = 1), half_im_matrix), axis = 1)
    half_dw_matrix = np.sum(w_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)

    return full_im_matrix, half_dw_matrix

def make_wake_im(wing: TangDowellWing, alpha: float, l_wake_c: int) -> np.ndarray:
    TE_wake_points = np.concatenate((wing.A[-1, :, :], wing.D[-1, -1, :][None, :]), axis = 0)
    wake_points = TE_wake_points + np.array([l_wake_c*wing.chord*np.cos(alpha), 0.0, l_wake_c*wing.chord*np.sin(alpha)])

    v_ind, w_ind = v_ind_by_vortex_ring(wing.panel_ctrl, wake_points[:-1], TE_wake_points[:-1], TE_wake_points[1:], wake_points[1:], gamma = 1)
    half_wake_im = np.sum(v_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
    full_wake_im = np.concatenate((np.flip(half_wake_im, axis = 1), half_wake_im), axis = 1)
    half_wake_dw = np.sum(w_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)

    return full_wake_im, half_wake_dw

def make_right_hand_side(wing: TangDowellWing, v_inf: float, alpha: float) -> np.ndarray:

    v_inf = (v_inf * np.array([np.cos(alpha), 0.0, np.sin(alpha)]))[None, :]
    rhs = np.sum(v_inf * wing.panel_normal.reshape(-1, 3), axis = 1, keepdims = False)
    rhs = np.concatenate((np.flip(rhs, axis = 0), rhs), axis = 0)

    return rhs

def generate_solutions(wing: TangDowellWing, v_inf: float, alpha: float, l_wake_c: Union[int, np.ndarray]) -> np.ndarray:

    te_panels_idx = (wing.n_c - 1)*wing.n_s
    bound_im, half_bound_dw = make_bound_im(wing)
    l_wake_c = np.asarray(l_wake_c, dtype = np.int64)

    results = []

    for length in l_wake_c:
        wake_im, wake_dw = make_wake_im(wing, alpha, length)
        full_im, half_dw = np.copy(bound_im), np.copy(half_bound_dw)
        full_im[:, 2*te_panels_idx:] += wake_im
        half_dw[:, te_panels_idx:] += wake_dw
        rhs = make_right_hand_side(wing, v_inf, alpha, length)

        gamma, info = gmres(full_im, rhs)

        if info != 0:
            raise RuntimeError(f"GMRES did not converge. Info code: {info}")
        gamma = gamma[gamma.size // 2:]
        solution = {'alpha': alpha, 'wake_length': length, 'gamma': gamma, 'downwash': half_dw @ gamma}
        results.append(solution)
    
    return results