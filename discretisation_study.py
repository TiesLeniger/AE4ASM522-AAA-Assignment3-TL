import matplotlib.pyplot as plt
import numpy as np

from src.wing import TangDowellWing
from src.vortex import *
from src.solver import *
from src.postprocessing import postprocessing

D2R = np.pi/180
R2D = 180/np.pi

# Define operating conditions
rho = 1.225                         # [kg/m^3], air density
v_inf = 20.0                        # [m/s], free stream velocity
alpha = 5.0 * D2R                   # [deg -> rad], angle of attack

chord_discr = [2, 3, 4, 5]
span_discr = range(10, 29)

CL_results = []
CD_results = []

for n_c in chord_discr:
    CL_subresults = []
    CD_subresults = []
    for n_s in span_discr:
        wing = TangDowellWing(n_c, n_s, "constant", "cosine")
        l_wake_c = 10*wing.chord

        # Generate solutions
        results = generate_solutions(wing, v_inf, alpha, l_wake_c)

        # Postprocess
        C_L, C_D = postprocessing(results, wing, rho, v_inf, make_plots = False)
        
        CL_subresults.append(C_L)
        CD_subresults.append(C_D)
    
    CL_results.append(CL_subresults)
    CD_results.append(CD_subresults)

# After filling CL_results and CD_results:
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

for i, n_c in enumerate(chord_discr):
    axs[0].plot(list(span_discr), CL_results[i], label=f'n_c = {n_c}')
    axs[1].plot(list(span_discr), CD_results[i], label=f'n_c = {n_c}')

xticks = list(span_discr)[::2]

axs[0].set_title('Lift Coefficient $C_L$ vs Spanwise Discretisation')
axs[0].set_xlabel('Number of spanwise panels ($n_s$)')
axs[0].set_ylabel('$C_L$')
axs[0].set_xticks(xticks)
axs[0].legend()
axs[0].grid(True)

axs[1].set_title('Drag Coefficient $C_D$ vs Spanwise Discretisation')
axs[1].set_xlabel('Number of spanwise panels ($n_s$)')
axs[0].set_xticks(xticks)
axs[1].set_ylabel('$C_D$')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()