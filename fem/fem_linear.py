import numpy as np

def get_constant_fem():
    """
    Default properties of a span-wise uniform beam.

    Beam properties taken from Table 1 in:
    Tang, D., & Dowell, E. H. (2002). Experimental and theoretical study of gust response for high-aspect-ratio wing. AIAA Journal, 40(3), 419–429.

    NOTE: KBT is set to 0.25*sqrt(EI*GJ) to be able to test strucural coupling effects. Set KBT = 0 to replicate the results from Tang and Dowell.
    """
    constant = {
        'm': 0.2351,  # beam mass per unit length, [kg/m]
        'Ip': 0.2056e-4,  # polar moment of inertia scaled with beam density per unit length (rho*Ip), [kgm**2/m]
        'd': -0.000508,  # distance from cross-sectional CG to the elastic (reference) axis, positive towards the trailing edge [m]
        'EI': 0.4186,  # bending stiffness, [Nm**2]
        'GJ': 0.9539,  # torsional stiffness, [Nm**2]
        'KBT': 0.1579,  # bend-twist coupling coefficient (KBT**2 < EI*GJ), [Nm**2]
        'zetaB': 0.02,  # structural damping in bending, cB/cB_crit, [-]
        'zetaT': 0.031,  # structural damping in torsion, cT/cT_crit, [-]
        'l': 0.4508,  # beam length, [m]
        'c': 0.0508,  # beam chord, [m]
        'n_el': 50,  # number of beam elements, [-]
        'bc': 'FF',  # boundary condition, FF -> free-free, CF -> clamped-free
    }

    return constant

def initialise_fem(const_fem):
    """
    Initialise the FEM model by generating the vectors of elemental properties.

    :param const_fem: dictionary of input data for the fem model
    :return:
    """

    # number of elements, nodes and DOFs:
    n_el = const_fem['n_el']
    n_nd = n_el + 1
    n_dof = n_nd*3

    # uniform beam properties:
    # ------------------------

    # Beam length:
    l = const_fem['l']

    # Stiffness properties:
    EI = const_fem['EI']*np.ones((n_el,))
    GJ = const_fem['GJ']*np.ones((n_el,))
    KBT = const_fem['KBT']*np.ones((n_el,))

    # Mass properties:
    m = const_fem['m']*np.ones((n_el,))
    Ip = const_fem['Ip']*np.ones((n_el,))
    d = const_fem['d']*np.ones((n_el,))

    # Nodal coordinates (uniformly spaced along the beam):
    y_nd = np.linspace(0, l, n_nd)

    # Element lengths:
    L = np.diff(y_nd)

    # Boundary conditions:
    bc = const_fem['bc']

    # Known DOFs:
    b_k = np.zeros((n_dof,), dtype=bool)
    if bc == 'FF':
        b_k[:] = False
    elif bc == 'CF':
        b_k[:3] = True

    # The unknown DOFs
    b_u = ~b_k

    # Data storage dictionary:
    fem = {
        'n_el': n_el,
        'n_nd': n_nd,
        'n_dof': n_dof,
        'EI': EI,
        'GJ': GJ,
        'KBT': KBT,
        'm': m,
        'Ip': Ip,
        'd': d,
        'y_nd': y_nd,
        'L': L,
        'b_k': b_k,
        'b_u': b_u
    }

    return fem

def initialise_dmo(const_dmo, fem):
    """

    :param const_dmo: dictionary of input data for the discrete mass objects
    :param fem: dictionary of FEM data
    :return:
    """
    l = fem['y_nd'][-1]
    n_dmo = len(const_dmo['ay'])

    dmo = None
    if n_dmo>0:

        y_dmo = np.array(const_dmo['ay'])*l
        m_dmo = np.array(const_dmo['m'])
        d_dmo = np.array(const_dmo['d'])
        S_dmo = m_dmo*d_dmo
        I_dmo = m_dmo*d_dmo**2 + np.array(const_dmo['I0'])

        dmo = {
            'n_dmo': n_dmo,  # number of discrte mass objects
            'y_dmo': y_dmo,  # spanwise position of discrete mass objects, [m]
            'm_dmo': m_dmo,  # mass of the discrete mass objects, [kg]
            'd_dmo': d_dmo,  # distance of the discrete mass object to the elastic axis, positive when the mass is towards the LE wrt. the elastic axis, [m]
            'S_dmo': S_dmo,  # fist moment of inertia wrt. to the elastic axis [kgm]
            'Ip_dmo': I_dmo  # moment of inertia wrt. the elastic axis [kgm**2]
        }

    return dmo

def elem_mass(l, m, Ip, d):
    """
    calculates an elemental mass matrix.

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

    :param l: element length, [m]
    :param d: distance from CG to elastic axis, positive towards the trailing edge [m]
    :param m: mass per unit length, [mkg/m]
    :param Ip: # polar moment of inertia scaled with beam density per unit length (rho*Ip), [kgm**2/m]
    :return mat_M: elemntal mass matrix [6 x 6]
    """
    mat_M = np.zeros((6, 6))

    mat_M[0, 0] = l*Ip/3.0
    mat_M[0, 1] = mat_M[1, 0] = 7.0*d*m*l/20.0
    mat_M[0, 2] = mat_M[2, 0] = d*m*l**2/20.0
    mat_M[0, 3] = mat_M[3, 0] = Ip*l/6.0
    mat_M[0, 4] = mat_M[4, 0] = 3.0*d*m*l/20.0
    mat_M[0, 5] = mat_M[5, 0] = -d*m*l**2/30.0

    mat_M[1, 1] = 13.0*l*m/35.0
    mat_M[1, 2] = mat_M[2, 1] = 11.0*m*l**2/210.0
    mat_M[1, 3] = mat_M[3, 1] = mat_M[0, 4]
    mat_M[1, 4] = mat_M[4, 1] = 9.0*m*l/70.0
    mat_M[1, 5] = mat_M[5, 1] = -13.0*m*l**2/420.0

    mat_M[2, 2] = m*l**3/105.0
    mat_M[2, 3] = mat_M[3, 2] = -mat_M[0, 5]
    mat_M[2, 4] = mat_M[4, 2] = -mat_M[1, 5]
    mat_M[2, 5] = mat_M[5, 2] = -m*l**3/140.0

    mat_M[3, 3] = mat_M[0, 0]
    mat_M[3, 4] = mat_M[4, 3] = mat_M[0, 1]
    mat_M[3, 5] = mat_M[5, 3] = -mat_M[0, 2]

    mat_M[4, 4] = mat_M[1, 1]
    mat_M[4, 5] = mat_M[5, 4] = -mat_M[1, 2]

    mat_M[5, 5] = m*l**3/105.0
    return mat_M

def mat_mass(fem, dmo = None):
    """
    Assembles the global mass matrix of the beam.

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
    :return MM: global mass matrix:
    """
    # Unpack FEM information
    # -----------------------
    L = fem['L']
    d = fem['d']
    m = fem['m']
    Ip = fem['Ip']

    n_el = fem['n_el']
    n_dof = fem['n_dof']

    b_u = fem['b_u']

    # Create global mass matrix:
    # --------------------------
    MM = np.zeros((n_dof, n_dof))
    for i in range(n_el):
        MM[3*i:3*(i + 2), 3*i:3*(i + 2)] += elem_mass(L[i], m[i], Ip[i], d[i])

    # Add discrete mass objects:
    # --------------------------
    if dmo is not None:
        # Unpack values:
        n_nd = fem['n_nd']
        y_nd = fem['y_nd']

        n_dmo = dmo['n_dmo']
        y_dmo = dmo['y_dmo']
        m_dmo = dmo['m_dmo']
        S_dmo = dmo['S_dmo']
        Ip_dmo = dmo['Ip_dmo']

        # indices of neighbouring nodes:
        idx2 = np.searchsorted(y_nd, y_dmo, 'left')
        idx1 = idx2-1

        for (i1, i2, y_i, m_i, S_i, Ip_i) in zip(idx1, idx2, y_dmo, m_dmo, S_dmo, Ip_dmo):

            # Find neighbouring nodes:
            y1 = y_nd[i1]
            y2 = y_nd[i2]

            # linearly split the values between the nodes based on the distance to the node:
            m1 = -(y_i - y2)/(y2 - y1)*m_i
            m2 = (y_i - y1)/(y2 - y1)*m_i

            S1 = -(y_i - y2)/(y2 - y1)*S_i
            S2 = (y_i - y1)/(y2 - y1)*S_i

            Ip1 = -(y_i - y2)/(y2 - y1)*Ip_i
            Ip2 = (y_i - y1)/(y2 - y1)*Ip_i

            # Add the values to the global mass matrix:

            MM[3*i1, 3*i1] += Ip1
            MM[3*i2, 3*i2] += Ip2

            MM[3*i1 + 1, 3*i1 + 1] += m1
            MM[3*i2 + 1, 3*i2 + 1] += m2

            MM[3*i1, 3*i1 + 1] += S1
            MM[3*i1 + 1, 3*i1] += S1
            MM[3*i2, 3*i2 + 1] += S2
            MM[3*i2 + 1, 3*i2] += S2

    # Apply boundary conditions:

    MM_red = MM[b_u, :][:, b_u]

    return MM_red

def elem_stiffness(l, EI, GJ, KBT):
    """
    calculates an elemental stiffness matrix.

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

    :param l: element length, [m]
    :param EI: bending stiffness, [Nm**2]
    :param GJ: torsional stiffness, [Nm**2]
    :param KBT: bend-twist coupling coefficient (KBT**2 < EI*GJ), [Nm**2]
    :return mat_K: element stiffness matrix [6 x 6]
    """
    mat_K = np.zeros((6, 6))

    mat_K[0, 0] = GJ/l
    mat_K[0, 2] = mat_K[2, 0] = KBT/l
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

def mat_stiffness(fem):
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
    # Unpack FEM information
    # -----------------------
    L = fem['L']
    EI = fem['EI']
    GJ = fem['GJ']
    KBT = fem['KBT']

    n_el = fem['n_el']
    n_dof = fem['n_dof']

    b_u = fem['b_u']

    # Create global mass matrix:
    # --------------------------
    KK = np.zeros((n_dof, n_dof))
    for i in range(n_el):
        KK[3*i:3*(i + 2), 3*i:3*(i + 2)] += elem_stiffness(L[i], EI[i], GJ[i], KBT[i])

    # Apply boundary conditions:
    # --------------------------
    KK_red = KK[b_u, :][:, b_u]

    return KK_red