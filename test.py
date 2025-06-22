from src.wing import TangDowellWing
from src.vortex import *
from src.solver import *
from src.postprocessing import postprocessing


point = np.array([[0.0, 2.0, 0.0]])
A = np.array([[1.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
B = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
C = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
D = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

print(v_ind_by_vortex_ring(point, A, B, C, D, gamma = 1))