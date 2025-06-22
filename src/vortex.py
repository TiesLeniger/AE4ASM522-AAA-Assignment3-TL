import numpy as np
from typing import Union

def v_ind_by_straight_filament(points: np.ndarray, startpoints: np.ndarray, endpoints: np.ndarray, gamma: Union[float, np.ndarray], threshold: float = 1e-6):
    points = points.reshape(-1, 3)[:, None, :]              # (N, 1, 3)
    startpoints = startpoints.reshape(-1, 3)[None, :, :]    # (1, M, 3)
    endpoints = endpoints.reshape(-1, 3)[None, :, :]        # (1, M, 3)
    N, M = points.shape[0], startpoints.shape[1]

    r0 = endpoints - startpoints                            # (1, M, 3)
    r1 = points - startpoints                               # (N, M, 3)
    r2 = points - endpoints                                 # (N, M, 3)
    norm_r1 = np.linalg.norm(r1, axis=2, keepdims=True)
    norm_r2 = np.linalg.norm(r2, axis=2, keepdims=True)

    # Avoid division by zero
    r1_singular = norm_r1 < threshold
    r2_singular = norm_r2 < threshold

    norm_r1 = np.where(r1_singular, 1.0, norm_r1)
    norm_r2 = np.where(r2_singular, 1.0, norm_r2)

    r1_cross_r2 = np.cross(r1, r2, axis=2)                  # (N, M, 3)
    norm_r1_cross_r2 = np.linalg.norm(r1_cross_r2, axis=2, keepdims=True)
    r1_cross_r2_singular = np.square(norm_r1_cross_r2) < threshold

    norm_r1_cross_r2 = np.where(r1_cross_r2_singular, 1.0, norm_r1_cross_r2)

    # (r1/|r1| - r2/|r2|)
    diff = r1 / norm_r1 - r2 / norm_r2                      # (N, M, 3)
    # Dot product with r0 (broadcast r0 to (N, M, 3))
    dot = np.sum(diff * r0, axis=2, keepdims=True)          # (N, M, 1)

    # Gamma shape: (M,) or (N, M) or scalar
    gamma = np.asarray(gamma)
    if gamma.ndim == 0:
        gamma = np.full((M,), gamma)
    if gamma.ndim == 1:
        gamma = gamma[None, :, None]                    # (1, M, 1)
    elif gamma.ndim == 2:                       
        gamma = gamma.flatten()[None, :, None]          # (1, M, 1)    

    v_induced = (gamma / (4 * np.pi)) * (r1_cross_r2 / (norm_r1_cross_r2 ** 2)) * dot  # (N, M, 3)
    
    mask = r1_singular | r2_singular | r1_cross_r2_singular
    v_induced[mask.repeat(3, axis = 2)] = 0.0

    return v_induced

def v_ind_by_vortex_ring(points: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, gamma: Union[float, np.ndarray], threshold: float = 1e-6, return_w: bool = False):

    v_AB = v_ind_by_straight_filament(points, A, B, gamma, threshold)
    v_BC = v_ind_by_straight_filament(points, B, C, gamma, threshold)
    v_CD = v_ind_by_straight_filament(points, C, D, gamma, threshold)
    v_DA = v_ind_by_straight_filament(points, D, A, gamma, threshold)

    if return_w:
        w_ind = v_AB + v_CD
        v_ind = w_ind + v_BC + v_DA
        return v_ind, w_ind
    else:
        return v_AB + v_BC + v_CD + v_DA