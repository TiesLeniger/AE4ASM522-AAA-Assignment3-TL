from src.wing import TangDowellWing
from src.vortex import *
from src.solver import *
from src.postprocessing import postprocessing

D2R = np.pi/180
R2D = 180/np.pi

# Instantiate wing
wing = TangDowellWing(4, 6, "constant", "cosine")

# Define operating conditions
rho = 1.225                         # [kg/m^3], air density
v_inf = 20.0                        # [m/s], free stream velocity
alpha = 5.0 * D2R                   # [deg -> rad], angle of attack
l_wake_c = 10*wing.chord            # [m], wake length w.r.t chord length

# Generate solutions
results = generate_solutions(wing, v_inf, alpha, l_wake_c)

# Postprocess
C_L, C_D = postprocessing(results, wing, rho, v_inf, make_plots = True)
print(C_L, C_D)