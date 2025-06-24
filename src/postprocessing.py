import numpy as np
import matplotlib.pyplot as plt
from src.wing import TangDowellWing
from pathlib import Path

def lift_at_panels(gamma: np.ndarray, v_inf: float, wing: TangDowellWing, rho: float) -> np.ndarray:

    gamma = gamma.reshape((wing.n_c, wing.n_s))                                             # Make sure gamma has the correct shape

    delta_Lij = np.zeros_like(gamma)                                                        # Initialize lift array
    delta_Lij[0, :] = rho*v_inf*gamma[0, :]*wing.panel_widths[0, :]                         # Lift on LE
    delta_Lij[1:, :] = rho*v_inf*(gamma[1:, :] - gamma[:-1, :])*wing.panel_widths[1:, :]    # Lift on other panels

    return delta_Lij

def drag_at_panels(gamma: np.ndarray, downwash: np.ndarray, wing: TangDowellWing, rho: float) -> np.ndarray:

    gamma = gamma.reshape((wing.n_c, wing.n_s))                                                     # Make sure gamma has the correct shape
    downwash = downwash.reshape((wing.n_c, wing.n_s))                                               # Make sure downwash has the correct shape

    delta_Dij = np.zeros_like(gamma)                                                                # Initialize lift array
    delta_Dij[0, :] = -rho*downwash[0, :]*gamma[0, :]*wing.panel_widths[0, :]                        # Lift on LE
    delta_Dij[1:, :] = -rho*downwash[1:, :]*(gamma[1:, :] - gamma[:-1, :])*wing.panel_widths[1:, :]  # Lift on other panels

    return delta_Dij

def spanwise_lift_distribution(gamma: np.ndarray, wing: TangDowellWing, v_inf: float) -> tuple[np.ndarray]:
    gamma = gamma.reshape((wing.n_c, wing.n_s))
    Clc = np.zeros_like(gamma)
    Clc[0, :] = 2*gamma[0, :]/v_inf
    Clc[1:, :] = 2*(gamma[1:, :] - gamma[:-1, :])/v_inf
    Clc = np.sum(Clc, axis = 0)
    Cl = Clc / wing.chord

    return Clc, Cl

def postprocessing(results: list, wing: TangDowellWing, rho: float, v_inf: float, make_plots: bool = True, save_plots: bool = False):

    C_L = []
    C_D = []

    if save_plots:
        path_to_output = Path.cwd() / "output"
    
    for result in results:
        # Unpack circulation
        gamma = (result['gamma']).reshape((wing.n_c, wing.n_s))

        # Lift
        delta_Lij = lift_at_panels(gamma, v_inf, wing, rho)
        L_tot = np.sum(delta_Lij)
        C_L.append(L_tot/(0.5*rho*v_inf*v_inf*wing.wing_area))

        # Drag
        delta_Dij = drag_at_panels(gamma, result['downwash'], wing, rho)
        D_tot = np.sum(delta_Dij)
        C_D.append(D_tot/(0.5*rho*v_inf*v_inf*wing.wing_area))

        if make_plots:
            # spanwise lift distribution
            Clc, Cl = spanwise_lift_distribution(gamma, wing, v_inf)
            plt.plot(wing.panel_cop[0, :, 1], Clc, color = 'red', label = r'$C_{l_c}$')
            plt.plot(wing.panel_cop[0, :, 1], Cl, color = 'blue', label = r'$C_l$')
            plt.xlabel('y [m]')
            plt.ylabel('Coefficient value [-]')
            plt.grid(True)
            plt.title("Spanwise lift distribution")
            plt.legend()
            if save_plots:
                save_path = path_to_output / f"spanwise_lift_dist_alpha_{result['alpha']}_lwake_{result['wake_length']}.png"
                plt.savefig(save_path)
            plt.show()

            # spanwise drag distribution
            Cd = np.sum(delta_Dij, axis = 0)
            Cd /= (0.5*rho*v_inf*wing.panel_widths[0, :]*wing.chord)
            plt.plot(wing.panel_cop[0, :, 1], Cd, color = 'red')
            plt.xlabel('y [m]')
            plt.ylabel(r"$C_d$ [-]")
            plt.grid(True)
            plt.title('Spanwise drag distribution')
            if save_plots:
                save_path = path_to_output / f"spanwise_drag_dist_alpha_{result['alpha']}_lwake_{result['wake_length']}.png"
                plt.savefig(save_path)
            plt.show()

    return C_L, C_D