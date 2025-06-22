import numpy as np
from scipy import linalg as ln

import FEM.fem_linear as fe
import FEM.fem_loads as fl

def ss_fem_lin(fem, dmo = None):
    """
    todo: Add the distribution matrix for discrete forces and combine it with the distribution matrix for distributed forces
    constructs the A and B matrices for the state-space model of the beam.

    the DOFs of the spate space model, xi, are organised as follows:

    xi ={xdot, x},

    where xdot is the first time derivative of the physical DOFs, x, which are organised as:

    ---------------------------------------------------
    ...
    ---------------------------------------------------
    theta_i -> torsional deflection at node i
    v_i     -> bending out-of plane deflection at node i
    beta_i  -> bending rotation at node i
    ---------------------------------------------------
    theta_i+1 -> torsional deflection at node i+1
    v_i+1     -> bending out-of plane deflection at node i+1
    beta_i+1  -> bending rotation at node i+1
    ---------------------------------------------------
    ...
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return A, Bdst: state-space system and the input matrix
    """
    MM = fe.mat_mass(fem, dmo)
    KK = fe.mat_stiffness(fem)
    DDdst = fl.mat_force_dst(fem)
    DDdsc = fl.mat_force_dsc(fem)

    n_dof_red = MM.shape[0]

    MMinv = ln.inv(MM)

    # System matrix:
    A = np.zeros((2*n_dof_red, 2*n_dof_red))
    A[0:n_dof_red, n_dof_red:2*n_dof_red] = -MMinv.dot(KK)
    A[n_dof_red:2*n_dof_red, 0:n_dof_red] = np.identity(n_dof_red)

    # Input matrix:
    Bdst = np.zeros((2*n_dof_red, n_dof_red))
    Bdst[0:n_dof_red, 0:n_dof_red] = MMinv.dot(DDdst)

    Bdsc = np.zeros((2*n_dof_red, n_dof_red))
    Bdsc[0:n_dof_red, 0:n_dof_red] = MMinv.dot(DDdsc)

    SS = {
        'A': A,  # system matrix
        'B_dst': Bdst,  # input matrix for distributed nodal forces ([N/m])
        'B_dsc': Bdsc,  # input matrix for discrete nodal forces (N)
    }

    return SS


def ss_fem_qsae_lin(U, fem, tsm, dmo = None):
    """
    todo: add the output matrices for aerodynamic loads
    constructs the A and B matrices for the state-space model of the beam.

    This model is using quasi-steady aerodynamic approximation

    The DOFs of the spate space model, x, are organised as follows:

    x ={xdot, x},

    where xdot is the first time derivative of the physical DOFs, x, which are organised as:

    ---------------------------------------------------
    ...
    ---------------------------------------------------
    theta_i -> torsional deflection at node i
    v_i     -> bending out-of plane deflection at node i
    beta_i  -> bending rotation at node i
    ---------------------------------------------------
    ...
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return AA, BB: state-space system and the input matrix
    """

    rho = tsm['rho']  # air density

    A = fe.mat_mass(fem, dmo)
    B = fl.mat_mass_aero(fem, tsm, usaero=False)
    D = fl.mat_damping_aero(fem, tsm, usaero=False)
    E = fe.mat_stiffness(fem)
    F = fl.mat_stiffness_aero(fem, tsm, usaero=False)

    n_dof_red = A.shape[0]

    n_tot = 2*n_dof_red

    M_inv = ln.inv((A + rho*B))

    # State-space system matrix:
    # --------------------------
    A_ss = np.zeros((n_tot, n_tot))
    A_ss[:n_dof_red, :n_dof_red] = (-1.)*M_inv.dot(rho*U*D)
    A_ss[:n_dof_red, n_dof_red:2*n_dof_red] = (-1.)*M_inv.dot(E + rho*U**2*F)
    A_ss[n_dof_red:2*n_dof_red, :n_dof_red] = np.identity(n_dof_red)

    # State-space input matrix static angle of attack:
    # ------------------------------------------------
    F_aoa0 = fl.vec_load_aoa0(fem, tsm)

    B_aoa0 = np.zeros((n_tot,))
    B_aoa0[:n_dof_red] = (-1.)*M_inv.dot(rho*U**2*F_aoa0)

    SS = {
        'A': A_ss,  # system matrix, A
        'B_aoa0': B_aoa0,  # input vector for the static angle of attack input
    }

    return SS


def ss_fem_qsae_lin2(U, fem, tsm, dmo = None):
    """
    todo: add the output matrices for aerodynamic loads
    constructs the A and B matrices for the state-space model of the beam.

    This model is using quasi-steady aerodynamic approximation

    The DOFs of the spate space model, x, are organised as follows:

    x ={xdot, x},

    where xdot is the first time derivative of the physical DOFs, x, which are organised as:

    ---------------------------------------------------
    ...
    ---------------------------------------------------
    theta_i -> torsional deflection at node i
    v_i     -> bending out-of plane deflection at node i
    beta_i  -> bending rotation at node i
    ---------------------------------------------------
    ...
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return AA, BB: state-space system and the input matrix
    """

    rho = tsm['rho']  # air density

    A = fe.mat_mass(fem, dmo)
    B = fl.mat_mass_aero2(fem, tsm, usaero=False)
    D = fl.mat_damping_aero2(fem, tsm, usaero=False)
    E = fe.mat_stiffness(fem)
    F = fl.mat_stiffness_aero2(fem, tsm, usaero=False)

    n_dof_red = A.shape[0]

    n_tot = 2*n_dof_red

    M_inv = ln.inv((A + rho*B))

    # State-space system matrix:
    # --------------------------
    A_ss = np.zeros((n_tot, n_tot))
    A_ss[:n_dof_red, :n_dof_red] = (-1.)*M_inv.dot(rho*U*D)
    A_ss[:n_dof_red, n_dof_red:2*n_dof_red] = (-1.)*M_inv.dot(E + rho*U**2*F)
    A_ss[n_dof_red:2*n_dof_red, :n_dof_red] = np.identity(n_dof_red)

    # State-space input matrix static angle of attack:
    # ------------------------------------------------
    F_aoa0 = fl.vec_load_aoa0(fem, tsm)

    B_aoa0 = np.zeros((n_tot,))
    B_aoa0[:n_dof_red] = (-1.)*M_inv.dot(rho*U**2*F_aoa0)

    SS = {
        'A': A_ss,  # system matrix, A
        'B_aoa0': B_aoa0,  # input vector for the static angle of attack input
    }

    return SS


def ss_fem_usae_lin(U, fem, tsm, dmo = None):
    """
    todo: Add the distribution matrix for discrete forces and combine it with the distribution matrix for distributed forces
    constructs the A and B matrices for the state-space model of the beam.

    the DOFs of the spate space model, x, are organised as follows:

    x ={xdot, x, w},

    where xdot is the first time derivative of the physical DOFs, x, which are organised as:

    ---------------------------------------------------
    ...
    ---------------------------------------------------
    theta_i -> torsional deflection at node i
    v_i     -> bending out-of plane deflection at node i
    beta_i  -> bending rotation at node i
    ---------------------------------------------------
    ...
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return AA, BB: state-space system and the input matrix
    """

    rho = tsm['rho']  # air density

    A = fe.mat_mass(fem, dmo)
    B = fl.mat_mass_aero(fem, tsm, usaero=True)
    D = fl.mat_damping_aero(fem, tsm, usaero=True)
    E = fe.mat_stiffness(fem)
    F = fl.mat_stiffness_aero(fem, tsm, usaero=True)
    W = fl.mat_influence_usaero(fem, tsm)
    W1 = fl.mat_state1_usaero(fem, tsm)
    W2 = fl.mat_state2_usaero(fem, tsm)

    n_dof_red = A.shape[0]
    n_w_red = W2.shape[0]

    n_tot = 2*n_dof_red + n_w_red

    M_inv = ln.inv((A + rho*B))

    # State-space system matrix:
    # --------------------------
    A_ss = np.zeros((n_tot, n_tot))
    A_ss[:n_dof_red, :n_dof_red] = (-1.)*M_inv.dot(rho*U*D)
    A_ss[:n_dof_red, n_dof_red:2*n_dof_red] = (-1.)*M_inv.dot(E + rho*U**2*F)
    A_ss[:n_dof_red, 2*n_dof_red:] = (-1.)*rho*U**3*M_inv.dot(W)
    A_ss[n_dof_red:2*n_dof_red, :n_dof_red] = np.identity(n_dof_red)
    A_ss[2*n_dof_red:, n_dof_red:2*n_dof_red] = W1
    A_ss[2*n_dof_red:, 2*n_dof_red:] = U*W2

    # State-space input matrix static angle of attack:
    # ------------------------------------------------
    F_aoa0 = fl.vec_load_aoa0(fem, tsm)

    B_aoa0 = np.zeros((n_tot,))
    B_aoa0[:n_dof_red] = (-1.)*M_inv.dot(rho*U**2*F_aoa0)

    SS = {
        'A': A_ss,  # system matrix, A
        'B_aoa0': B_aoa0,  # input vector for the static angle of attack input
    }

    return SS


def beam_ae_lin(x, U, fem, tsm, dmo = None):
    """
    todo: Add the distribution matrix for discrete forces and combine it with the distribution matrix for distributed forces
    constructs the A and B matrices for the state-space model of the beam.

    the DOFs of the spate space model, x, are organised as follows:

    x ={xdot, x, w},

    where xdot is the first time derivative of the physical DOFs, x, which are organised as:

    ---------------------------------------------------
    ...
    ---------------------------------------------------
    theta_i -> torsional deflection at node i
    v_i     -> bending out-of plane deflection at node i
    beta_i  -> bending rotation at node i
    ---------------------------------------------------
    ...
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return AA, BB: state-space system and the input matrix
    """

    rho = tsm['rho']  # air density

    A = fe.mat_mass(fem, dmo)
    B = fl.mat_mass_aero(fem, tsm, usaero=True)
    D = fl.mat_damping_aero(fem, tsm, usaero=True)
    E = fe.mat_stiffness(fem)
    F = fl.mat_stiffness_aero(fem, tsm, usaero=True)
    W = fl.mat_influence_usaero(fem, tsm)
    W1 = fl.mat_state1_usaero(fem, tsm)
    W2 = fl.mat_state2_usaero(fem, tsm)

    n_dof_red = A.shape[0]
    n_w_red = W2.shape[0]

    n_tot = 2*n_dof_red + n_w_red

    M_inv = ln.inv((A + rho*B))

    Q = np.zeros((n_tot, n_tot))
    Q[:n_dof_red, :n_dof_red] = (-1.)*M_inv.dot(rho*U*D)
    Q[:n_dof_red, n_dof_red:2*n_dof_red] = (-1.)*M_inv.dot(E + rho*U**2*F)
    Q[:n_dof_red, 2*n_dof_red:] = (-1.)*rho*U**3*M_inv.dot(W)
    Q[n_dof_red:2*n_dof_red, :n_dof_red] = np.identity(n_dof_red)
    Q[2*n_dof_red:, n_dof_red:2*n_dof_red] = W1
    Q[2*n_dof_red:, 2*n_dof_red:] = U*W2

    xprime = Q.dot(x)

    return xprime


def beam_qsae_lin(x, U, fem, tsm, dmo = None):
    """
    todo: Add the distribution matrix for discrete forces and combine it with the distribution matrix for distributed forces
    constructs the A and B matrices for the state-space model of the beam.

    This model is using quasi-steady aerodynamic approximation

    the DOFs of the spate space model, x, are organised as follows:

    x ={xdot, x},

    where xdot is the first time derivative of the physical DOFs, x, which are organised as:

    ---------------------------------------------------
    ...
    ---------------------------------------------------
    theta_i -> torsional deflection at node i
    v_i     -> bending out-of plane deflection at node i
    beta_i  -> bending rotation at node i
    ---------------------------------------------------
    ...
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return AA, BB: state-space system and the input matrix
    """

    rho = tsm['rho']  # air density

    A = fe.mat_mass(fem, dmo)
    B = fl.mat_mass_aero(fem, tsm, usaero=False)
    D = fl.mat_damping_aero(fem, tsm, usaero=False)
    E = fe.mat_stiffness(fem)
    F = fl.mat_stiffness_aero(fem, tsm, usaero=False)

    n_dof_red = A.shape[0]

    n_tot = 2*n_dof_red

    M_inv = ln.inv((A + rho*B))

    Q = np.zeros((n_tot, n_tot))
    Q[:n_dof_red, :n_dof_red] = (-1.)*M_inv.dot(rho*U*D)
    Q[:n_dof_red, n_dof_red:2*n_dof_red] = (-1.)*M_inv.dot(E + rho*U**2*F)
    Q[n_dof_red:2*n_dof_red, :n_dof_red] = np.identity(n_dof_red)

    xprime = Q.dot(x)

    return xprime


def beam_qsae_lin2(x, U, fem, tsm, dmo = None):
    """
    todo: Add the distribution matrix for discrete forces and combine it with the distribution matrix for distributed forces
    constructs the A and B matrices for the state-space model of the beam.

    This model is using quasi-steady aerodynamic approximation

    the DOFs of the spate space model, x, are organised as follows:

    x ={xdot, x},

    where xdot is the first time derivative of the physical DOFs, x, which are organised as:

    ---------------------------------------------------
    ...
    ---------------------------------------------------
    theta_i -> torsional deflection at node i
    v_i     -> bending out-of plane deflection at node i
    beta_i  -> bending rotation at node i
    ---------------------------------------------------
    ...
    ---------------------------------------------------

    :param fem: dictionary with discretised FE model parameters
    :return AA, BB: state-space system and the input matrix
    """

    rho = tsm['rho']  # air density

    A = fe.mat_mass(fem, dmo)
    B = fl.mat_mass_aero2(fem, tsm, usaero=False)
    D = fl.mat_damping_aero2(fem, tsm, usaero=False)
    E = fe.mat_stiffness(fem)
    F = fl.mat_stiffness_aero2(fem, tsm, usaero=False)

    n_dof_red = A.shape[0]

    n_tot = 2*n_dof_red

    M_inv = ln.inv((A + rho*B))

    Q = np.zeros((n_tot, n_tot))
    Q[:n_dof_red, :n_dof_red] = (-1.)*M_inv.dot(rho*U*D)
    Q[:n_dof_red, n_dof_red:2*n_dof_red] = (-1.)*M_inv.dot(E + rho*U**2*F)
    Q[n_dof_red:2*n_dof_red, :n_dof_red] = np.identity(n_dof_red)

    xprime = Q.dot(x)

    return xprime
