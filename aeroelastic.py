import numpy as np
from scipy.sparse.linalg import gmres

from src.wing import TangDowellWing
from src.fembeam import FEMBeam
from src.vortex import *
from src.solver import *
from src.postprocessing import *

# Constants
D2R = np.pi/180
R2D = 180/np.pi

# Operating conditions
v_inf = 20.0                # [m/s], freestream velocity
rho = 1.221                 # [kg/m^3], air density
alpha = 5.0 * D2R           # [deg -> rad], angle of attack
l_wake_c = 30               # [-]

# Instantiate wing
wing = TangDowellWing(4, 16, "constant", "cosine")

# Make a displacement vector
xi = np.zeros((wing.fem.n_dof,))
gamma_prev = np.zeros((wing.n_c*wing.n_s,))

# Iteration parameters
it = 0
max_iter = 100
tol = 1e-6

# Influence matrices
bound_im, bound_dw = make_bound_im(wing)
wake_im, wake_dw = make_wake_im(wing, alpha, l_wake_c)
te_panels_idx = (wing.n_c - 1)*wing.n_s
full_im, full_dw = np.copy(bound_im), np.copy(bound_dw)
full_im[:, te_panels_idx:] += wake_im
full_dw[:, te_panels_idx:] += wake_dw

while it < max_iter:
    rhs = make_right_hand_side(wing, v_inf, alpha, xi, wing.T_as)
    gamma, info = gmres(full_im, rhs)

    if np.linalg.norm(gamma - gamma_prev) < tol:
        break

    gamma_prev = gamma.copy()

    downwash = full_dw @ gamma
    delta_Lij = lift_at_panels(gamma, v_inf, wing, rho)
    delta_Dij = drag_at_panels(gamma, downwash, wing, rho)

    r, f, q = wing.map_aero_to_displ(delta_Lij, alpha)
    xi = wing.fem.calculate_displacement(r, f, q)
    xi = xi.reshape(-1, 3)
    xi = (np.concatenate((np.flip(xi, axis = 0), xi), axis = 0)).flatten()