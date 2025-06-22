import numpy as np
from scipy.sparse.linalg import gmres
from vortex import *
from wing import TangDowellWing

def make_influence_matrix(wing: TangDowellWing, alpha: float, l_wake_c: int) -> np.ndarray:
    
    def _make_bound_influence_matrix(wing) -> np.ndarray:
        v_ind, w_ind = v_ind_by_vortex_ring(wing.panel_ctrl, wing.A, wing.B, wing.C, wing.D, gamma = 1, return_w=True)
        half_im_matrix = np.sum(v_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
        full_im_matrix = np.concatenate((np.flip(half_im_matrix, axis = 1), half_im_matrix), axis = 1)
        half_dw_matrix = np.sum(w_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
        full_dw_matrix = np.concatenate((np.flip(half_dw_matrix, axis = 1), half_dw_matrix), axis = 1)

        return full_im_matrix, full_dw_matrix
    
    def _make_wake_influence_matrix(wing, alpha, l_wake_c) -> np.ndarray:

        TE_wake_points = np.concatenate((wing.A[-1, :, :], wing.D[-1, -1, :][None, :]), axis = 0)
        wake_points = TE_wake_points + np.array([l_wake_c*wing.chord*np.cos(alpha), 0.0, l_wake_c*wing.chord*np.sin(alpha)])

        v_ind, w_ind = v_ind_by_vortex_ring(wing.panel_ctrl, wake_points[:-1], TE_wake_points[:-1], TE_wake_points[1:], wake_points[1:], gamma = 1)
        half_wake_im = np.sum(v_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
        full_wake_im = np.concatenate((np.flip(half_wake_im, axis = 1), half_wake_im), axis = 1)
        half_wake_dw = np.sum(w_ind * (wing.panel_normal.reshape(-1,3))[:, None, :], axis = 2, keepdims = False)
        full_wake_dw = np.concatenate((np.flip(half_wake_dw, axis = 1), half_wake_dw), axis = 1)

        return full_wake_im, full_wake_dw
    
    influence_matrix, downwash_matrix = _make_bound_influence_matrix(wing)
    wake_im, wake_dw = _make_wake_influence_matrix(wing, alpha, l_wake_c)
    te_panels_idx = (wing.n_c - 1)*wing.n_s
    influence_matrix[:, te_panels_idx:] += wake_im
    downwash_matrix[:, te_panels_idx:] += wake_dw

    return influence_matrix, downwash_matrix

def make_right_hand_side(wing: TangDowellWing, v_inf: float, alpha: float) -> np.ndarray:

    v_inf = (v_inf * np.array([np.cos(alpha), 0.0, np.sin(alpha)]))[None, :]
    rhs = np.sum(v_inf * wing.panel_normal.reshape(-1, 3), axis = 1, keepdims = False)
    rhs = np.concatenate((np.flip(rhs, axis = 0), rhs), axis = 0)

    return rhs

def generate_solution(wing: TangDowellWing, v_inf: float, alpha: float, l_wake_c: float) -> np.ndarray:

    im, dw = make_influence_matrix(wing, alpha, l_wake_c)
    rhs = make_right_hand_side(wing, v_inf, alpha, l_wake_c)

    gamma, info = gmres(im, rhs)

    if info == 0:
        print("Solution converged.")
    else:
        raise RuntimeError(f"GMRES did not converge. Info code: {info}")
    
    return gamma, dw