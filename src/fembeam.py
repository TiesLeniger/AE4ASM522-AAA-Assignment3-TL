import numpy as np
import json
from pathlib import Path

class FEMBeam:

    def __init__(self, filename: str):
        # Construct path object to input file
        path_to_file = Path.cwd() / "input" / filename
        
        # Check filetype
        if not path_to_file.suffix:
            path_to_file = path_to_file.with_suffix('.json')
        elif path_to_file.suffix.lower() != ".json":
            raise ValueError(f"Expected JSON file, got {path_to_file.suffix}")
        
        # Read json data into a dict and assign attributes to FEMBeam class
        with open(path_to_file) as file:
            json_dict = json.load(file)
        
        # Initialise FEM properties
        fem_dict = json_dict['fem']
        self._init_fem_properties(fem_dict)

        # Initialse DMO
        if "dmo" in json_dict and json_dict['dmo']:                     # Check if it is present and non empty
            dmo_dict = json_dict['dmo']
            self._init_dmo_properties(dmo_dict)
        else:
            self.n_dmo = 0

        self.stiffness_matrix()                                     # Initialise stiffness matrix
        self.discrete_force_matrix()                                # Initialise discrete force matrix
        self.distributed_force_matrix()                             # Initialise distributed force matrix

    def _init_fem_properties(self, fem_dict: dict):
        self.n_el = fem_dict['n_el']
        self.n_nd = self.n_el + 1
        self.n_dof = self.n_nd*3
        self.l = fem_dict['l']
        self.EI = fem_dict['EI']*np.ones((self.n_el,))
        self.GJ = fem_dict['GJ']*np.ones((self.n_el,))
        self.KbT = fem_dict['KBT']*np.ones((self.n_el,))
        self.m = fem_dict['m']*np.ones((self.n_el,))
        self.Ip = fem_dict['Ip']*np.ones((self.n_el,))
        self.d = fem_dict['d']*np.ones((self.n_el,))
        self.y_nd = np.linspace(0, self.l, self.n_nd)
        self.L_el = np.diff(self.y_nd)
        self.bc = fem_dict['bc']
        b_k = np.zeros((self.n_dof,), dtype = bool)
        if self.bc == "CF":
            b_k[:3] = True
        self.b_k = b_k
        self.b_u = ~b_k

    def _init_dmo_properties(self, dmo_dict: dict):
        self.n_dmo = len(dmo_dict['ay'])
        self.y_dmo = np.array(dmo_dict['ay'])*self.l
        self.m_dmo = np.array(dmo_dict['m'])
        self.d_dmo = np.array(dmo_dict['d'])
        self.S_dmo = self.m_dmo*self.d_dmo
        self.I_dmo = self.m_dmo*self.d_dmo**2 + np.array(dmo_dict['I0'])

    def stiffness_matrix(self) -> np.ndarray:
        """
        Assembles the global stiffness matrix of the beam.

        The organisation of the DOFs is:
        ---------------------------------------------------
        theta_i -> torsional deflection at node i
        v_i     -> bending out-of plane deflection at node i
        beta_i  -> bending rotation at node i
        ---------------------------------------------------
        theta_i+1 -> torsional deflection at node i+1
        v_i+1     -> bending out-of plane deflection at node i+1
        beta_i+1  -> bending rotation at node i+1
        ---------------------------------------------------

        :param fem: dictionary with discretised FE model parameters
        :return KK: global stiffness matrix:
        """
        def _element_stiffness_matrix(l: float, EI: float, GJ: float, KbT: float) -> np.ndarray:
            mat_K = np.zeros((6, 6))

            mat_K[0, 0] = GJ/l
            mat_K[0, 2] = mat_K[2, 0] = KbT/l
            mat_K[0, 3] = mat_K[3, 0] = -mat_K[0, 0]
            mat_K[0, 5] = mat_K[5, 0] = -mat_K[0, 2]

            mat_K[1, 1] = 12.0*EI/l**3
            mat_K[1, 2] = mat_K[2, 1] = 6.0*EI/l**2
            mat_K[1, 4] = mat_K[4, 1] = -mat_K[1, 1]
            mat_K[1, 5] = mat_K[5, 1] = mat_K[1, 2]

            mat_K[2, 2] = 4.0*EI/l
            mat_K[2, 3] = mat_K[3, 2] = -mat_K[0, 2]
            mat_K[2, 4] = mat_K[4, 2] = -mat_K[1, 2]
            mat_K[2, 5] = mat_K[5, 2] = 2*EI/l

            mat_K[3, 3] = mat_K[0, 0]
            mat_K[3, 5] = mat_K[5, 3] = mat_K[0, 2]

            mat_K[4, 4] = mat_K[1, 1]
            mat_K[4, 5] = mat_K[5, 4] = -mat_K[1, 2]

            mat_K[5, 5] = mat_K[2, 2]

            return mat_K

        # Create global mass matrix:
        KK = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_el):
            KK[3*i:3*(i + 2), 3*i:3*(i + 2)] += _element_stiffness_matrix(self.L_el[i], self.EI[i], self.GJ[i], self.KbT[i])

        # Apply boundary conditions:
        self.KK_red = KK[self.b_u, :][:, self.b_u]

    def distributed_force_matrix(self) -> np.ndarray:
        """
        Assembles the global distributed force matrix for the beam.

        This matrix is used to map distributed loads per unit length (e.g., gravity, aerodynamic lift, or distributed torque, shear, and bending moment)
        defined along the beam span to equivalent nodal forces and moments in the finite element model.

        The organisation of the DOFs is:
        ---------------------------------------------------
        theta_i   -> torsional deflection at node i
        v_i       -> bending out-of-plane deflection at node i
        beta_i    -> bending rotation at node i
        ---------------------------------------------------
        theta_i+1 -> torsional deflection at node i+1
        v_i+1     -> bending out-of-plane deflection at node i+1
        beta_i+1  -> bending rotation at node i+1
        ---------------------------------------------------

        The resulting reduced matrix (self.DD_red) can be used as:
            load_vector = self.DD_red @ distributed_load_vector

        :return: None. The reduced distributed force matrix is stored as self.DD_red.
        """
        def _element_distributed_force_matrix(l):

            mat_D = np.zeros((6, 6))

            mat_D[0, 0] = l/3.0
            mat_D[0, 3] = l/6.0

            mat_D[1, 1] = 7.0*l/20
            mat_D[1, 2] = -0.5
            mat_D[1, 4] = 3.0*l/20
            mat_D[1, 5] = -0.5

            mat_D[2, 1] = l**2/20
            mat_D[2, 2] = l/12.0
            mat_D[2, 4] = l**2/30
            mat_D[2, 5] = -l/12.0

            mat_D[3, 0] = l/6.0
            mat_D[3, 3] = l/3.0

            mat_D[4, 1] = 3.0*l/20.0
            mat_D[4, 2] = 0.5
            mat_D[4, 4] = 7.0*l/20.0
            mat_D[4, 5] = 0.5

            mat_D[5, 1] = -l**2/30.0
            mat_D[5, 2] = -l/12.0
            mat_D[5, 4] = -l**2/20
            mat_D[5, 5] = l/12.0

            return mat_D

        DD = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_el):
            DD[3*i:3*(i + 2), 3*i:3*(i + 2)] += _element_distributed_force_matrix(self.L_el[i])

        self.dst_DD_red = DD[self.b_u, :][:, self.b_u]

    def discrete_force_matrix(self) -> np.ndarray:
        """
        Assembles the global discrete force matrix for the beam.

        This matrix is used to map discrete nodal loads (such as concentrated torques, shear forces, or bending moments applied directly at the nodes)
        to the corresponding degrees of freedom (DOFs) in the finite element model.

        The organisation of the DOFs is:
        ---------------------------------------------------
        theta_i   -> torsional deflection at node i
        v_i       -> bending out-of-plane deflection at node i
        beta_i    -> bending rotation at node i
        ---------------------------------------------------
        theta_i+1 -> torsional deflection at node i+1
        v_i+1     -> bending out-of-plane deflection at node i+1
        beta_i+1  -> bending rotation at node i+1
        ---------------------------------------------------

        The resulting reduced matrix (DD_red) can be used as:
            load_vector = DD_red @ nodal_load_vector

        :return: The reduced discrete force matrix (DD_red), which maps nodal loads to the unknown DOFs.
        """
        DD = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_nd):
            DD[3*i:3*(i + 1), 3*i:3*(i + 1)] += np.identity(3)

        self.dsc_DD_red = DD[self.b_u, :][:, self.b_u]

    def generate_load_vector(self, r: np.ndarray, f: np.ndarray, q: np.ndarray) -> np.ndarray:

        """
        This function distributes the loads to an appropriate DOF.

        The input is an array of nodal values of torque, shear force, and bending moment. The size of these arrays must
        therefore be (n_nd,).

        NOTE:
        The inputs can be concentrated loads or distributed loads. Depending on the type of the load the correct element
        force matrix must be used.

        :param fem: dictionary with FEM properties
        :param r: distributed torque at nodes, [Nm or Nm/m], shape (n_nd,)
        :param f: distributed shear force at nodes, [N or N/m], shape (n_nd,)
        :param q: distributed bending moment at nodes, [Nm or Nm/m], shape (n_nd,)
        :return vec_load_red: load vector at unknown DOFs
        """
        vec_load = np.zeros((self.n_dof,))
        vec_load[0::3] = r
        vec_load[1::3] = f
        vec_load[2::3] = q

        vec_load_red = vec_load[self.b_u]

        return vec_load_red
    
    def calculate_displacement(self, r: np.ndarray, f: np.ndarray, q: np.ndarray) -> np.ndarray:

        f_weight = np.zeros(3*self.n_el)
        f_weight[1::3] = -9.81 * self.m
        structural_weight_load = self.dst_DD_red @ f_weight
        
        external_loads = self.generate_load_vector(r, f, q)

        load_vector = structural_weight_load + external_loads

        if self.n_dmo > 0:
            dmo_nodal_forces = np.zeros((self.n_el, 3))
            for i in range(self.n_dmo):
                node_idx = np.argmin(np.abs(self.y_nd - self.y_dmo[i]))
                dmo_nodal_forces[node_idx-1, 1] += -self.m_dmo[i] * 9.81
            dmo_vec = dmo_nodal_forces.flatten()
            dmo_load_red = self.dsc_DD_red @ dmo_vec
            load_vector += dmo_load_red

        u_red = np.linalg.solve(self.KK_red, load_vector)

        u_full = np.zeros(self.n_dof)
        u_full[self.b_u] = u_red

        return u_full