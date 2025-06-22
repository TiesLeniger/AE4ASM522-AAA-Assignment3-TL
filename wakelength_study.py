import matplotlib.pyplot as plt

from vlm.wing import TangDowellWing
from vlm.vortex import *
from vlm.solver import *
from vlm.postprocessing import postprocessing

D2R = np.pi/180

# Instantiate wing
wing = TangDowellWing(4, 21, "constant", "cosine")

# Define operating conditions
rho = 1.225                         # [kg/m^3], air density
v_inf = 20.0                        # [m/s], free stream velocity
alpha = 5.0 * D2R                   # [deg -> rad], angle of attack

l_wake_c = list(range(1, 31))

results = generate_solutions(wing, v_inf, alpha, l_wake_c)

CL, CD = postprocessing(results, wing, rho, v_inf, make_plots = False)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(l_wake_c, CL, color='blue')
axs[0].set_title('Lift Coefficient $C_L$ vs Wake Length')
axs[0].set_xlabel('Wake Length / Chord ($l_{wake}/c$)')
axs[0].set_ylabel('$C_L$')
axs[0].grid(True)

axs[1].plot(l_wake_c, CD, color='red')
axs[1].set_title('Drag Coefficient $C_D$ vs Wake Length')
axs[1].set_xlabel('Wake Length / Chord ($l_{wake}/c$)')
axs[1].set_ylabel('$C_D$')
axs[1].grid(True)

plt.tight_layout()
plt.show()