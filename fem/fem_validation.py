import numpy as np


def beam_FF_modes_bending(constant):
    """
    Calculates the first 5 eigen frequencies and modes of a free-free beam in pure bending with constant mass and
    stiffness properties along the span of the beam

    # See example solution for Free-Free Beam on page 88/89 in Introduction to Structural Dynamics and Aeroelasticity
    (Hodges and Pierce 2011, 2nd edition)

    :param constant: dictionary of beam properties
    :return
        omega: eigen frequencies of the beam
        x: spanwise location where the modeshape is evaluated
        phi: modal displacement evaluated at x
    """

    # Beam properties:
    EI, m, l = constant['EI'], constant['m'], constant['l']

    # The first 5 solutions of the characteristic equation 3.276 on p 88
    aL = np.array([4.73004, 7.85320, 10.9956, 14.1372, 17.2788])

    alpha = aL/l

    # Eigen frequencies
    omega = alpha**2*np.sqrt(EI/m)

    # eigen modes:
    y = np.linspace(0, l, 100)
    beta = (np.cosh(aL) - np.cos(aL))/(np.sinh(aL) - np.sin(aL))

    phi = []
    for i in range(len(aL)):
        a_i, b_i = alpha[i], beta[i]

        c, ch = np.cos(a_i*y), np.cosh(a_i*y)
        s, sh = np.sin(a_i*y), np.sinh(a_i*y)

        phi_i = (c + ch) - b_i*(s + sh)

        phi.append(phi_i)

    return omega, y, phi


def beam_CF_modes_bending(constant):
    """
    Calculates the first 5 eigen frequencies and modes of a clamped-free beam in pure bending with constant mass and
    stiffness properties along the span of the beam

    # See example solution for Free-Free Beam on page 82/83 in Introduction to Structural Dynamics and Aeroelasticity
    (Hodges and Pierce 2011, 2nd edition)

    :param constant: dictionary of beam properties
    :return
        omega: eigen frequencies of the beam
        x: spanwise location where the modeshape is evaluated
        phi: modal displacement evaluated at x
    """

    # Beam properties:
    EI, m, l = constant['EI'], constant['m'], constant['l']

    # The first 5 solutions of the characteristic equation 3.256 on p 83
    aL = np.array([1.87510, 4.69409, 7.85476, 10.9955, 14.1372])

    alpha = aL/l

    # Eigen frequencies
    omega = alpha**2*np.sqrt(EI/m)

    # eigen modes:
    y = np.linspace(0, l, 100)
    beta = (np.cosh(aL) + np.cos(aL))/(np.sinh(aL) + np.sin(aL))

    phi = []
    for i in range(len(aL)):
        a_i, b_i = alpha[i], beta[i]

        c, ch = np.cos(a_i*y), np.cosh(a_i*y)
        s, sh = np.sin(a_i*y), np.sinh(a_i*y)

        phi_i = (ch - c) - b_i*(sh - s)

        phi.append(phi_i)

    return omega, y, phi


def beam_FF_modes_torsion(constant):
    """
    Calculates the first 5 eigen frequencies and modes of a free-free beam in pure TORSION with constant mass and
    stiffness properties along the span of the beam

    # See example solution for Free-Free Beam on page 64/65 in Introduction to Structural Dynamics and Aeroelasticity
    (Hodges and Pierce 2011, 2nd edition)

    :param constant: dictionary of beam properties
    :return
        omega: eigen frequencies of the beam
        y: spanwise location where the modeshape is evaluated
        phi: modal displacement evaluated at x
    """

    # Beam properties:
    GJ, Ip, l = constant['GJ'], constant['Ip'], constant['l']

    # The first 5 solutions of the characteristic equation 3.177 on p 65
    aL = np.arange(1, 6)*np.pi

    alpha = aL/l

    # Eigen frequencies
    omega = alpha*np.sqrt(GJ/Ip)

    # eigen modes:
    y = np.linspace(0, l, 100)

    phi = []
    for i in range(len(aL)):
        a_i = alpha[i]
        phi_i = np.cos(a_i*y)

        phi.append(phi_i)

    return omega, y, phi


def beam_CF_modes_torsion(constant):
    """
    Calculates the first 5 eigen frequencies and modes of a clamped-free beam in pure torsion with constant mass and
    stiffness properties along the span of the beam

    # See example solution for Free-Free Beam on page 82/83 in Introduction to Structural Dynamics and Aeroelasticity
    (Hodges and Pierce 2011, 2nd edition)

    :param constant: dictionary of beam properties
    :return
        omega: eigen frequencies of the beam
        x: spanwise location where the modeshape is evaluated
        phi: modal displacement evaluated at x
    """

    # Beam properties:
    GJ, Ip, l = constant['GJ'], constant['Ip'], constant['l']

    # The first 5 solutions of the characteristic equation 3.256 on p 83
    aL = (2*np.arange(1, 6) - 1)*np.pi/2

    alpha = aL/l

    # Eigen frequencies
    omega = alpha*np.sqrt(GJ/Ip)

    # eigen modes:
    y = np.linspace(0, l, 100)

    phi = []
    for i in range(len(aL)):
        a_i = alpha[i]
        phi_i = np.sin(a_i*y)

        phi.append(phi_i)

    return omega, y, phi


def beam_CF_constant_distributed_load(q, r, constant):
    """
    Caclulates a bendiding and torsional response to a constant distributed shear force,[N/m], and torque, [Nm/m], for a
    uniform clamped beam of constant material and cross-sectional properties with no bend-twist coupling.

    :param q: magnitude of constant distributed shear force, [N/m]
    :param r: magnitude of constant distributed torque, [Nm/m]
    :param constant:  dictionary of beam properties
    :return:
        y: spanwise location where the modeshape is evaluated
        w: out-of-plane bending deformation [m]
        theta: torsional deformation [rad]
    """
    # Beam properties:
    EI, GJ, l = constant['EI'], constant['GJ'], constant['l']

    y = np.linspace(0, l, 100)

    w = q*y**2*(6*l**2 - 4*l*y + y**2)/(24*EI)
    theta = r*(2*l*y - y**2)/(2*GJ)

    return y, w, theta


def beam_CF_constant_tip_load(V, T, constant):
    """
    Caclulates a bendiding and torsional response to a tip shear force [N] and torque,[Nm], for a uniform clamped beam
    of constant material and cross-sectional properties with no bend-twist coupling.

    :param V: tip shear force, [N]
    :param T: tip torque, [Nm]
    :param constant:  dictionary of beam properties
    :return:
        y: spanwise location where the modeshape is evaluated
        w: out-of-plane bending deformation [m]
        theta: torsional deformation [rad]
    """
    # Beam properties:
    EI, GJ, l = constant['EI'], constant['GJ'], constant['l']

    y = np.linspace(0, l, 100)

    w = V*y**2*(3*l - y)/(6*EI)
    theta = T*y/GJ

    return y, w, theta
