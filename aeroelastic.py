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
alpha = 3.0 * D2R           # [deg -> rad], angle of attack
l_wake_c = 30               # [-]

# Instantiate wing
wing = TangDowellWing(4, 20, "constant", "cosine")

def aeroelastic_analysis(wing: TangDowellWing, v_inf: float, rho: float, alpha: float, l_wake_c: float, max_iter: int = 100, tol: float = 1e-6):

    # Make a displacement vector
    xi = np.zeros((wing.fem.n_dof,))
    gamma_prev = np.zeros((wing.n_c*wing.n_s,))

    # Iteration parameters
    it = 0
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
        
        # gamma, info = gmres(full_im, rhs)
        gamma = np.linalg.solve(full_im, rhs)

        if np.linalg.norm(gamma - gamma_prev) < tol:
            break

        gamma_prev = gamma.copy()

        delta_Lij = lift_at_panels(gamma, v_inf, wing, rho)

        r, f, q = wing.map_aero_to_displ(delta_Lij, alpha)
        xi = wing.fem.calculate_displacement(r, f, q)

        it += 1

    return gamma, xi

# region question 1

# KbT = [-0.1, 0.0, 0.1]
# gamma_results = []
# xi_results = []
# for coeff in KbT:
#     wing.fem.KbT = coeff*np.sqrt(wing.fem.EI*wing.fem.GJ)
#     wing.fem.stiffness_matrix()                                     # Initialise stiffness matrix
#     wing.fem.discrete_force_matrix()                                # Initialise discrete force matrix
#     wing.fem.distributed_force_matrix()                             # Initialise distributed force matrix
#     gamma, xi = aeroelastic_analysis(wing, v_inf, rho, alpha, l_wake_c)
#     gamma_results.append(gamma)
#     xi_results.append(xi)

# for i in range(3):
#     Clc, Cl = spanwise_lift_distribution(gamma_results[i], wing, v_inf)
#     plt.plot(wing.panel_cop[0, :, 1], Cl, label = f"Bend-twist coupling {KbT[i]:.1f}")
# plt.xlabel('y [m]')
# plt.ylabel(r'$C_l$ [-]')
# plt.grid(True)
# plt.title(fr"Spanwise lift distribution VLM-FEM coupling $\alpha$ = {alpha*R2D:.1f} [deg]")
# plt.legend()
# plt.show()

# fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
# for j in range(3):
#     axs[0].plot(wing.fem.y_nd, xi_results[j][0::3]*R2D, label=rf"Bend-twist coupling {KbT[j]:.1f}")
#     axs[1].plot(wing.fem.y_nd, xi_results[j][1::3], label=f"Bend-twist coupling {KbT[j]:.1f}")
#     axs[2].plot(wing.fem.y_nd, xi_results[j][2::3]*R2D, label=f"Bend-twist coupling {KbT[j]:.1f}")

# axs[0].set_ylabel('Torsion [deg]')
# axs[0].legend()
# axs[0].grid(True)

# axs[1].set_ylabel('Bending [m]')
# axs[1].legend()
# axs[1].grid(True)

# axs[2].set_xlabel('y [m]')
# axs[2].set_ylabel('Rotation [deg]')
# axs[2].legend()
# axs[2].grid(True)

# plt.suptitle(fr'Spanwise structural displacements $\alpha$ = {alpha*R2D:.1f} [deg]')
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# endregion

# region question 2

# gamma_rigid, _ = aeroelastic_analysis(wing, v_inf, rho, alpha, l_wake_c, max_iter=1)
# gamma_flex, _ = aeroelastic_analysis(wing, v_inf, rho, alpha, l_wake_c)
# Clc_r, Cl_r = spanwise_lift_distribution(gamma_rigid, wing, v_inf)
# CL_r = np.sum(lift_at_panels(gamma_rigid, v_inf, wing, rho))/(0.5*rho*v_inf*v_inf*wing.wing_area)
# Clc_f, Cl_f = spanwise_lift_distribution(gamma_flex, wing, v_inf)
# CL_f = np.sum(lift_at_panels(gamma_flex, v_inf, wing, rho))/(0.5*rho*v_inf*v_inf*wing.wing_area)
# plt.plot(wing.panel_cop[0, :, 1], Cl_r, color = 'red', label = f"rigid")
# plt.plot(wing.panel_cop[0, :, 1], Cl_f, color = 'blue', label = f"flexible")
# plt.xlabel('y [m]')
# plt.ylabel(r'$C_l$ [-]')
# plt.grid(True)
# plt.title(fr"Spanwise lift distribution $\alpha$ = {alpha*R2D:.1f} [deg]")
# plt.legend()
# plt.show()

# print(CL_r, CL_f)

# endregion

# region question 3
gamma, xi = aeroelastic_analysis(wing, v_inf, rho, alpha, l_wake_c, max_iter = 1)

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axs[0].plot(wing.fem.y_nd, xi[0::3]*R2D, label=r'$\theta$ (torsion)')
axs[1].plot(wing.fem.y_nd, xi[1::3], label=r'$v$ (bending)')
axs[2].plot(wing.fem.y_nd, xi[2::3]*R2D, label=r'$\beta$ (rotation)')

axs[0].set_ylabel('Torsion [deg]')
axs[0].legend()
axs[0].grid(True)

axs[1].set_ylabel('Bending [m]')
axs[1].legend()
axs[1].grid(True)

axs[2].set_xlabel('y [m]')
axs[2].set_ylabel('Rotation [deg]')
axs[2].legend()
axs[2].grid(True)

plt.suptitle(fr'Spanwise structural displacements $\alpha$ = {alpha*R2D:.1f} [deg]')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# endregion