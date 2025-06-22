from src.wing import TangDowellWing
from src.vortex import *
from src.solver import *
from src.postprocessing import postprocessing

wing = TangDowellWing(6, 18, "constant", "cosine")

wing.plot_wing()