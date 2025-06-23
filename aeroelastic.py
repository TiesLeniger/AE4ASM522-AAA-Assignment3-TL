import numpy as np

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

# Instantiate wing
wing = TangDowellWing(4, 16, "constant", "cosine")