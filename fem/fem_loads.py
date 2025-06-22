import numpy as np

# region Distribution matrices
def elem_force_dst(l):
    """
    calculates an elemental distribution matrix for distributed loads, such as torque, [Nm/m], shear force, [N/m], and
    bending moment, [Nm/m], distribution.

    The organisation of the generalized loads is:
    ---------------------------------------------------
    r_i  -> value of torque distribution at node i, [Nm/m]
    f_i  -> value of shear force distribution at node i, [N/m]
    q_i  -> value of bending moment distribution at node i, [Nm/m]
    ---------------------------------------------------
    r_i+1  -> value of torque distribution at node i, [Nm/m]
    f_i+1  -> value of shear force distribution at node i, [N/m]
    q_i+1  -> value of bending moment distribution at node i, [Nm/m]
    ---------------------------------------------------

    :param l: element length, [m]
    :return mat_D: elemental matrix of generalised forces
    """

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

def elem_force_dsc():
    """
    calculates an elemental distribution matrix for discrete nodal generalized loads, such as torque, [Nm], shear force, [N], and
    bending moment, [Nm].

    The organisation of the generalized loads is:
    ---------------------------------------------------
    T_i  -> value of torque  at node i, [Nm]
    V_i  -> value of shear force at node i, [N]
    M_i  -> value of bending moment at node i, [Nm]
    ---------------------------------------------------
    T_i+1  -> value of torque at node i, [Nm]
    V_i+1  -> value of shear force at node i, [N]
    M_i+1  -> value of bending moment at node i, [Nm]
    ---------------------------------------------------

    :return mat_D: elemental matrix of generalized forces
    """

    mat_D = np.identity(3)

    return mat_D

def mat_force_dst(fem):
    """
    Assembles the global matrix of distributed generalised forces.

    The organisation of the generalized loads is:
    ----------------------------------------------------------------
    r_i  -> value of torque distribution at node i, [Nm/m]
    f_i  -> value of shear force distribution at node i, [N/m]
    q_i  -> value of bending moment distribution at node i, [Nm/m]
    ----------------------------------------------------------------
    r_i+1  -> value of torque distribution at node i, [Nm/m]
    f_i+1  -> value of shear force distribution at node i, [N/m]
    q_i+1  -> value of bending moment distribution at node i, [Nm/m]
    ----------------------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return DD: global stiffness matrix:
    """
    # Unpack FEM information
    # -----------------------
    L = fem['L']

    n_el = fem['n_el']
    n_dof = fem['n_dof']

    b_u = fem['b_u']

    # Create global mass matrix:
    # --------------------------
    DD = np.zeros((n_dof, n_dof))
    for i in range(n_el):
        DD[3*i:3*(i + 2), 3*i:3*(i + 2)] += elem_force_dst(L[i])

    DD_red = DD[b_u, :][:, b_u]

    return DD_red

def mat_force_dsc(fem):
    """
    Assembles the global matrix of discrete generalised forces.

    The organisation of the generalized loads is:
    ---------------------------------------------------
    T_i  -> value of torque  at node i, [Nm]
    V_i  -> value of shear force at node i, [N]
    M_i  -> value of bending moment at node i, [Nm]
    ---------------------------------------------------
    T_i+1  -> value of torque at node i, [Nm]
    V_i+1  -> value of shear force at node i, [N]
    M_i+1  -> value of bending moment at node i, [Nm]
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return DD: global stiffness matrix:
    """
    # Unpack FEM information
    # -----------------------
    n_nd = fem['n_nd']
    n_dof = fem['n_dof']

    b_u = fem['b_u']

    # Create global mass matrix:
    # --------------------------
    DD = np.zeros((n_dof, n_dof))
    for i in range(n_nd):
        DD[3*i:3*(i + 1), 3*i:3*(i + 1)] += elem_force_dsc()

    DD_red = DD[b_u, :][:, b_u]

    return DD_red

def generate_load_vector(fem, r, f, q):
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

    n_dof = fem['n_dof']
    b_u = fem['b_u']

    vec_load = np.zeros((n_dof,))
    vec_load[0::3] = r
    vec_load[1::3] = f
    vec_load[2::3] = q

    vec_load_red = vec_load[b_u]

    return vec_load_red
# endregion

# region Strip theory 2DOF (heave, pitch)

# def initialise_aero(const_tsm, fem):
#     # Unpack dicts:
#     # -------------

#     # Aerodynamic properties:
#     ay = np.array(const_tsm['ay'])
#     c = np.array(const_tsm['c'])
#     ac = np.array(const_tsm['ac'])

#     rho = const_tsm['rho']

#     # FEM properties:
#     y_nd = fem['y_nd']
#     b_k = fem['b_k']
#     n_nd = fem['n_nd']
#     # n_dof = fem['n_dof']

#     # Interpolate to nodal positions:
#     y = y_nd[-1]*ay

#     c_nd = np.interp(y_nd, y, c)
#     b_nd = c_nd/2

#     ac_nd = np.interp(y_nd, y, ac)
#     xf_nd = ac_nd*c_nd
#     a_nd = xf_nd/b_nd - 1

#     # Number of lag states:
#     #----------------------
#     # There are 2 lag states per theta_i and v_i DOF (=> 2*2*n_nd).
#     # There is no lag state for beta_i DOF since the aerodynamic force does not depend on it
#     n_w = 2*2*n_nd

#     # Known lag states (must be omitted due to boundary conditions):
#     mask = np.zeros(b_k.shape, dtype=bool)
#     mask[2::3] = True
#     b_k_red = b_k[~mask]

#     bw_k = np.zeros((n_w,), dtype=bool)
#     for i in range(2*n_nd):
#         bw_k[2*i:2*(i + 1)] = b_k_red[i]

#     bw_u = ~bw_k

#     # Data storage dictionary:
#     tsm = {
#         'n_w': n_w,  # number of all lag states (2*n_dof), [-]
#         'c_nd': c_nd,  # chord at FEM nodes, [m]
#         'b_nd': b_nd,  # semi-chord at FEM nodes, [m]
#         'ac_nd': ac_nd,  # distance from the leading edge to the beam axis in chords at FEM nodes, [-]
#         'xf_nd': xf_nd,  # distance from the leading edge to the beam axis at FEM nodes, [m]
#         'a_nd': a_nd,  # distance to the beam axis measured from the mid-chord point in semi-chords, (xf/b -1), [-]
#         'rho': rho,  # air density, [kg/m**3]
#         'bw_k': bw_k,  # map of known lag states (due to boundary conditions), [-]
#         'bw_u': bw_u,  # map of unknown lag states (due to boundary conditions), [-]
#     }

#     return tsm


# def map_str2ae():
#     """
#     Map structural DOFs of an element, (theta, v, beta)_i, to aerodynamic DOFs of the strip (h, theta)_i

#     Note:
#     > h = -v
#     > theta = theta

#     :return: T1, mapping matrix
#     """
#     T1 = np.zeros((2, 3))
#     T1[0, 1] = -1
#     T1[1, 0] = 1

#     return T1


# def map_ae2str():
#     """
#     Map aerodynamic lift and moment,(l, m_xf), to structural torsional moment and shear force, (r, f).

#     Note:
#     T2 = T1**T

#     :return: T2, mapping matrix
#     """

#     T2 = np.zeros((3, 2))
#     T2[0, 1] = 1
#     T2[1, 0] = -1

#     return T2


# def map_inf2inf():
#     """
#     Reshuffle the order of influence coefficients to match the order in A.27 on p.562 (Dimitriadis, 2017):
#     w' = [w1'(theta), w2'(theta, w3'(v), w4'(v)] => w = [w1(h), w2(h, w3'(theta), w4'(theta)]

#     NOTE: the mapping matrix is of size (4 x 6), because the lag states are also assigned to the bending slope DOF
#     beta = (dv/dy), which doesn't have any effect on the aerodynamic forces. This is done in order to maintain a regular
#     number of lag states (2 per structural DOF).

#     :return: T3, mapping matrix
#     """

#     T3 = np.zeros((4, 4))
#     T3[0, 2] = 1
#     T3[1, 3] = 1
#     T3[2, 0] = 1
#     T3[3, 1] = 1

#     return T3


# def elem_mass_aero2(prop_aero, prop_elem, usaero=True):
#     """
#     todo: add docstring
#     :param prop_aero:
#     :return:
#     """

#     L = prop_elem['L']

#     T1 = map_str2ae()
#     T2 = map_ae2str()

#     if usaero:
#         B = ae.matrix_mass2DOF_usaero(prop_aero)
#     else:
#         B = ae.matrix_mass2DOF_qsaero(prop_aero)

#     # Change the order of aerodynamic loads to match order of DOFs in the FEM and the positivie direction definitions (h = -v)
#     B = T2.dot(B.dot(T1))

#     BBelem = np.array([
#         [B[0, 0]*L/3, 7*B[0, 1]*L/20, B[0, 1]*L**2/20, B[0, 0]*L/6, 3*B[0, 1]*L/20, -B[0, 1]*L**2/30],
#         [7*B[1, 0]*L/20, 13*B[1, 1]*L/35, 11*B[1, 1]*L**2/210, 3*B[1, 0]*L/20, 9*B[1, 1]*L/70, -13*B[1, 1]*L**2/420],
#         [B[1, 0]*L**2/20, 11*B[1, 1]*L**2/210, B[1, 1]*L**3/105, B[1, 0]*L**2/30, 13*B[1, 1]*L**2/420, -B[1, 1]*L**3/140],
#         [B[0, 0]*L/6, 3*B[0, 1]*L/20, B[0, 1]*L**2/30, B[0, 0]*L/3, 7*B[0, 1]*L/20, -B[0, 1]*L**2/20],
#         [3*B[1, 0]*L/20, 9*B[1, 1]*L/70, 13*B[1, 1]*L**2/420, 7*B[1, 0]*L/20, 13*B[1, 1]*L/35, -11*B[1, 1]*L**2/210],
#         [-B[1, 0]*L**2/30, -13*B[1, 1]*L**2/420, -B[1, 1]*L**3/140, -B[1, 0]*L**2/20, -11*B[1, 1]*L**2/210, B[1, 1]*L**3/105]
#     ])

#     return BBelem


# def mat_mass_aero2(fem, tsm, usaero=True):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_el = fem['n_el']
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']
#     L = fem['L']

#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))

#     BB = np.zeros((n_dof, n_dof))
#     for i in range(n_el):
#         prop_aero = {
#             'a': 0.5*(a_nd[i] + a_nd[i + 1]),
#             'b': 0.5*(b_nd[i] + b_nd[i + 1]),
#             'ch': 0.5*(ch_nd[i] + ch_nd[i + 1]),
#         }

#         prop_elem = {
#             'L': L[i]
#         }

#         BBelem = elem_mass_aero2(prop_aero, prop_elem, usaero=usaero)

#         BB[3*i:3*(i + 2), 3*i:3*(i + 2)] += BBelem

#     # omit known DOFs:
#     BB_red = BB[b_u, :][:, b_u]

#     return BB_red

# def elem_damping_aero2(prop_aero, prop_elem, usaero=True):
#     """
#     todo: add docstring
#     :param prop_aero:
#     :return:
#     """

#     L = prop_elem['L']

#     T1 = map_str2ae()
#     T2 = map_ae2str()

#     if usaero:
#         D = ae.matrix_damping2DOF_usaero(prop_aero)
#     else:
#         D = ae.matrix_damping2DOF_qsaero(prop_aero)

#     # Change the order of aerodynamic loads to match order of DOFs in the FEM and the positivie direction definitions (h = -v)
#     D = T2.dot(D.dot(T1))

#     DDelem = np.array([
#         [D[0, 0]*L/3, 7*D[0, 1]*L/20, D[0, 1]*L**2/20, D[0, 0]*L/6, 3*D[0, 1]*L/20, -D[0, 1]*L**2/30],
#         [7*D[1, 0]*L/20, 13*D[1, 1]*L/35, 11*D[1, 1]*L**2/210, 3*D[1, 0]*L/20, 9*D[1, 1]*L/70, -13*D[1, 1]*L**2/420],
#         [D[1, 0]*L**2/20, 11*D[1, 1]*L**2/210, D[1, 1]*L**3/105, D[1, 0]*L**2/30, 13*D[1, 1]*L**2/420, -D[1, 1]*L**3/140],
#         [D[0, 0]*L/6, 3*D[0, 1]*L/20, D[0, 1]*L**2/30, D[0, 0]*L/3, 7*D[0, 1]*L/20, -D[0, 1]*L**2/20],
#         [3*D[1, 0]*L/20, 9*D[1, 1]*L/70, 13*D[1, 1]*L**2/420, 7*D[1, 0]*L/20, 13*D[1, 1]*L/35, -11*D[1, 1]*L**2/210],
#         [-D[1, 0]*L**2/30, -13*D[1, 1]*L**2/420, -D[1, 1]*L**3/140, -D[1, 0]*L**2/20, -11*D[1, 1]*L**2/210, D[1, 1]*L**3/105]
#     ])

#     return DDelem


# def mat_damping_aero2(fem, tsm, usaero=True):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_el = fem['n_el']
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']
#     L = fem['L']

#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))

#     DD = np.zeros((n_dof, n_dof))
#     for i in range(n_el):
#         prop_aero = {
#             'a': 0.5*(a_nd[i] + a_nd[i + 1]),
#             'b': 0.5*(b_nd[i] + b_nd[i + 1]),
#             'ch': 0.5*(ch_nd[i] + ch_nd[i + 1]),
#         }

#         prop_elem = {
#             'L': L[i]
#         }

#         DDelem = elem_damping_aero2(prop_aero, prop_elem, usaero=usaero)

#         DD[3*i:3*(i + 2), 3*i:3*(i + 2)] += DDelem

#     # omit known DOFs:
#     DD_red = DD[b_u, :][:, b_u]

#     return DD_red


# def elem_stiffness_aero2(prop_aero, prop_elem, usaero=True):
#     """
#     todo: add docstring
#     :param prop_aero:
#     :return:
#     """

#     L = prop_elem['L']

#     T1 = map_str2ae()
#     T2 = map_ae2str()

#     if usaero:
#         F = ae.matrix_stiffness2DOF_usaero(prop_aero)
#     else:
#         F = ae.matrix_stiffness2DOF_qsaero(prop_aero)

#     # Change the order of aerodynamic loads to match the order of DOFs in the FEM and the positive direction definitions (h = -v)
#     F = T2.dot(F.dot(T1))

#     FFelem = np.array([
#         [F[0, 0]*L/3, 7*F[0, 1]*L/20, F[0, 1]*L**2/20, F[0, 0]*L/6, 3*F[0, 1]*L/20, -F[0, 1]*L**2/30],
#         [7*F[1, 0]*L/20, 13*F[1, 1]*L/35, 11*F[1, 1]*L**2/210, 3*F[1, 0]*L/20, 9*F[1, 1]*L/70, -13*F[1, 1]*L**2/420],
#         [F[1, 0]*L**2/20, 11*F[1, 1]*L**2/210, F[1, 1]*L**3/105, F[1, 0]*L**2/30, 13*F[1, 1]*L**2/420, -F[1, 1]*L**3/140],
#         [F[0, 0]*L/6, 3*F[0, 1]*L/20, F[0, 1]*L**2/30, F[0, 0]*L/3, 7*F[0, 1]*L/20, -F[0, 1]*L**2/20],
#         [3*F[1, 0]*L/20, 9*F[1, 1]*L/70, 13*F[1, 1]*L**2/420, 7*F[1, 0]*L/20, 13*F[1, 1]*L/35, -11*F[1, 1]*L**2/210],
#         [-F[1, 0]*L**2/30, -13*F[1, 1]*L**2/420, -F[1, 1]*L**3/140, -F[1, 0]*L**2/20, -11*F[1, 1]*L**2/210, F[1, 1]*L**3/105]
#     ])

#     return FFelem


# def mat_stiffness_aero2(fem, tsm, usaero=True):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_el = fem['n_el']
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']
#     L = fem['L']

#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))

#     FF = np.zeros((n_dof, n_dof))
#     for i in range(n_el):
#         # take the average of the nodal values to get an element-wise value:
#         prop_aero = {
#             'a': 0.5*(a_nd[i] + a_nd[i + 1]),
#             'b': 0.5*(b_nd[i] + b_nd[i + 1]),
#             'ch': 0.5*(ch_nd[i] + ch_nd[i + 1]),
#         }

#         prop_elem = {
#             'L': L[i]
#         }

#         FFelem = elem_stiffness_aero2(prop_aero, prop_elem, usaero=usaero)

#         FF[3*i:3*(i + 2), 3*i:3*(i + 2)] += FFelem

#     # omit known DOFs:
#     FF_red = FF[b_u, :][:, b_u]

#     return FF_red


# def elem_mass_aero(prop_aero, usaero=True):
#     """
#     todo: add docstring
#     :param prop_aero:
#     :return:
#     """

#     T1 = map_str2ae()
#     T2 = map_ae2str()

#     if usaero:
#         B_aero = ae.matrix_mass2DOF_usaero(prop_aero)
#     else:
#         B_aero = ae.matrix_mass2DOF_qsaero(prop_aero)

#     return T2.dot(B_aero.dot(T1))


# def mat_mass_aero(fem, tsm, usaero=True):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']

#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))

#     BB = np.zeros((n_dof, n_dof))
#     for i in range(b_nd.shape[0]):
#         prop_aero = {
#             'a': a_nd[i],
#             'b': b_nd[i],
#             'ch': ch_nd[i],
#         }

#         BB[3*i:3*(i + 1), 3*i:3*(i + 1)] = elem_mass_aero(prop_aero, usaero=usaero)

#     # omit known DOFs:
#     BB_red = BB[b_u, :][:, b_u]

#     # Account for distributed forces:
#     DDdst = mat_force_dst(fem)

#     return DDdst.dot(BB_red)


# def elem_damping_aero(prop_aero, usaero=True):
#     """
#     todo: add docstring

#     :param prop_aero:
#     :return:
#     """
#     T1 = map_str2ae()
#     T2 = map_ae2str()

#     if usaero:
#         D_aero = ae.matrix_damping2DOF_usaero(prop_aero)
#     else:
#         D_aero = ae.matrix_damping2DOF_qsaero(prop_aero)

#     return T2.dot(D_aero.dot(T1))


# def mat_damping_aero(fem, tsm, usaero=True):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']

#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))

#     DD = np.zeros((n_dof, n_dof))
#     for i in range(b_nd.shape[0]):
#         prop_aero = {
#             'a': a_nd[i],
#             'b': b_nd[i],
#             'ch': ch_nd[i],
#         }

#         DD[3*i:3*(i + 1), 3*i:3*(i + 1)] = elem_damping_aero(prop_aero, usaero=usaero)

#     # omit known DOFs:
#     DD_red = DD[b_u, :][:, b_u]

#     # Account for distributed forces:
#     DDdst = mat_force_dst(fem)

#     return DDdst.dot(DD_red)


# def elem_stiffness_aero(prop_aero, usaero=True):
#     """
#     todo: add docstring

#     :param prop_aero:
#     :return:
#     """
#     T1 = map_str2ae()
#     T2 = map_ae2str()

#     if usaero:
#         F_aero = ae.matrix_stiffness2DOF_usaero(prop_aero)
#     else:
#         F_aero = ae.matrix_stiffness2DOF_qsaero(prop_aero)

#     return T2.dot(F_aero.dot(T1))


# def mat_stiffness_aero(fem, tsm, usaero=True):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']

#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))

#     FF = np.zeros((n_dof, n_dof))
#     for i in range(b_nd.shape[0]):
#         prop_aero = {
#             'a': a_nd[i],
#             'b': b_nd[i],
#             'ch': ch_nd[i],
#         }

#         FF[3*i:3*(i + 1), 3*i:3*(i + 1)] = elem_stiffness_aero(prop_aero, usaero=usaero)

#     # omit known DOFs:
#     FF_red = FF[b_u, :][:, b_u]

#     # Account for distributed forces:
#     DDdst = mat_force_dst(fem)

#     return DDdst.dot(FF_red)


# def elem_influence_usaero(prop_aero):
#     """
#     todo: add docstring

#     :param prop_aero:
#     :return:
#     """
#     T2 = map_ae2str()
#     T3 = map_inf2inf()

#     W_aero = ae.matrix_aero2DOF_influence(prop_aero)

#     return T2.dot(W_aero.dot(T3))


# def mat_influence_usaero(fem, tsm):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']

#     n_w = tsm['n_w']
#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))
#     bw_u = tsm['bw_u']

#     WW = np.zeros((n_dof, n_w))
#     for i in range(n_nd):
#         prop_aero = {
#             'a': a_nd[i],
#             'b': b_nd[i],
#             'ch': ch_nd[i],
#         }

#         WW[3*i:3*(i + 1), 4*i:4*(i + 1)] = elem_influence_usaero(prop_aero)

#     # omit known DOFs:
#     WW_red = WW[b_u, :][:, bw_u]

#     # Account for distributed forces:
#     DDdst = mat_force_dst(fem)

#     return DDdst.dot(WW_red)


# def elem_state1_usaero():
#     """
#     todo: add docstring

#     :return:
#     """
#     T1 = map_str2ae()
#     T3inv = np.transpose(map_inf2inf())

#     W1_aero = ae.matrix_aero2DOF_state1()

#     return T3inv.dot(W1_aero.dot(T1))


# def mat_state1_usaero(fem, tsm):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_nd = fem['n_nd']
#     n_dof = fem['n_dof']
#     b_u = fem['b_u']

#     n_w = tsm['n_w']
#     bw_u = tsm['bw_u']

#     WW1 = np.zeros((n_w, n_dof))
#     for i in range(n_nd):
#         WW1[4*i:4*(i + 1), 3*i:3*(i + 1)] = elem_state1_usaero()

#     # omit known DOFs:
#     WW1_red = WW1[bw_u, :][:, b_u]

#     return WW1_red


# def elem_state2_usaero(prop_aero):
#     """
#     todo: add docstring

#     :return:
#     """
#     T3 = map_inf2inf()
#     T3inv = np.transpose(T3)

#     W2_aero = ae.matrix_aero2DOF_state2(prop_aero)

#     return T3inv.dot(W2_aero.dot(T3))


# def mat_state2_usaero(fem, tsm):
#     """
#     todo: add docstring

#     :param fem:
#     :param tsm:
#     :return:
#     """
#     n_nd = fem['n_nd']

#     n_w = tsm['n_w']
#     b_nd = tsm['b_nd']
#     a_nd = tsm['a_nd']
#     ch_nd = np.zeros((n_nd,))
#     bw_u = tsm['bw_u']

#     WW2 = np.zeros((n_w, n_w))
#     for i in range(n_nd):
#         prop_aero = {
#             'a': a_nd[i],
#             'b': b_nd[i],
#             'ch': ch_nd[i],
#         }

#         WW2[4*i:4*(i + 1), 4*i:4*(i + 1)] = elem_state2_usaero(prop_aero)

#     # omit known DOFs:
#     WW2_red = WW2[bw_u, :][:, bw_u]

#     return WW2_red


# def vec_load_aoa0(fem, tsm):
#     """
#     :param fem:
#     :param tsm: 
#     :return: 
#     """
#     FF_red = mat_stiffness_aero(fem, tsm, usaero=False)

#     # Distribution vector:
#     T_aoa0_red = np.zeros((FF_red.shape[0],))
#     T_aoa0_red[::3] = 1

#     return FF_red.dot(T_aoa0_red)
#endregion