# -*- coding: utf-8 -*-
# Copyright 2019-2020, David J. Pine
#
# This package is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This package is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this package. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
# Suppress runtime warnings for innocuous zero-divides
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def g_PY(r, phi):
    """
    Calculates the hard-sphere radial distribution function using the Percus-Yevick
    closure scheme.

    Parameters
    ----------
    phi : float
        volume fraction of spheres
    r :  float or numpy.ndarray
        radial distance divided by hard-sphere diameter

    Returns
    -------
    g(r) : float or numpy.ndarray
        radial distribution function for hard spheres using the the Percus-Yevick
        closure scheme.

    References
    ----------
    A. Trokhymchuk, I. Nezbeda, J. Jirsák, & D. Henderson, Hard-sphere radial distribution
    function again, J. Chem. Phys. 123, 024501, (2005)

    A. Trokhymchuk, I. Nezbeda, J. Jirsák, & D. Henderson, Erratum: "Hard-sphere radial
    distribution function again", J. Chem. Phys. 124, 149902, (2006)

    Notes:
    -----
    The original paper gives the wrong formulas for gamma and mu; the correct formulas
    are in the erratum.

    Authors:
    Wenhai Zheng
    David Pine
    """
    # Calculate d
    det = (3. * (3. + phi * (6. + phi * (1. + phi * (-2. + phi))))) ** 0.5
    d = (2. * phi * (phi ** 2. - 3. * phi - 3. + det)) ** (1. / 3.)  # Eq (32)
    # Calculate parameters depending only on phi and d
    alpha0 = 2. * phi / (1. - phi) * (-1. + 0.25 * d / phi - 0.5 * phi / d)  # Appendix
    beta0 = 2. * phi / (1. - phi) * np.sqrt(3) * (-0.25 * d / phi - 0.5 * phi / d)  # Appendix
    mu = 2. * phi / (1. - phi) * (-1. - 0.5 * d / phi + phi / d)  # Eq (29) in Erratum
    # Calculate gamma (Eq (30) in Erratum), which depends on phi, alpha0, beta0, and mu (& d)
    aa, bb = alpha0 * alpha0, beta0 * beta0
    num1 = (alpha0 * (aa + bb) - mu * (aa - bb)) * (1. + 0.5 * phi) + (aa + bb - mu * alpha0) * (1 + 2 * phi)
    den1 = (aa + bb - 2. * mu * alpha0) * (1. + 0.5 * phi) - mu * (1. + 2. * phi)
    gamma = np.arctan(-num1 / (den1 * beta0))

    # Calculate parameters depending only on phi
    g_expt = (0.25 / phi) * (
            (1. + phi * (1. + phi * (1. - (2. * phi / 3.) * (1. + phi)))) / (1. - phi) ** 3 - 1.)  # Eq (25)
    r_min = 2.0116 + phi * (-1.0647 + 0.0538 * phi)  # Eq (35)
    g_min = 1.0286 + phi * (-0.6095 + phi * (3.5781 + phi * (-21.3651 + phi * (42.6344 - 33.8485 * phi))))  # Eq (36)
    omega = -0.682 * np.exp(-24.697 * phi) + 4.720 + 4.450 * phi  # Appendix
    kappa = 4.674 * np.exp(-3.935 * phi) + 3.536 * np.exp(-56.270 * phi)  # Appendix
    alpha = 44.554 + 79.868 * phi + 116.432 * phi * phi - 44.652 * np.exp(2. * phi)  # Appendix
    beta = -5.022 + 5.857 * phi + 5.089 * np.exp(-4. * phi)  # Appendix
    # Calculate parameters b, a, delta, and c : Eqs (21) - (24)
    num2 = (g_min - (g_expt / r_min) * np.exp(mu * (r_min - 1.))) * r_min
    den2 = np.cos(beta * (r_min - 1.) + gamma) * np.exp(alpha * (r_min - 1.)) - np.cos(gamma) * np.exp(
        mu * (r_min - 1.))
    b = num2 / den2
    a = g_expt - b * np.cos(gamma)
    delta = -omega * r_min - np.arctan((kappa * r_min + 1.) / (omega * r_min))
    c = r_min * (g_min - 1.) * np.exp(kappa * r_min) / np.cos(omega * r_min + delta)
    # Depletion & structural parts of g(r) : Eqs (14)-(16)
    g_dep = a / r * np.exp(mu * (r - 1.)) + b / r * np.cos(beta * (r - 1.) + gamma) * np.exp(alpha * (r - 1.))
    g_str = 1. + c / r * np.cos(omega * r + delta) * np.exp(-kappa * r)
    g_outside = np.where(r <= r_min, g_dep, g_str)

    return np.where(r >= 1, g_outside, 0.)


def s_PY(qd, phi):
    """
    Calculates the hard-sphere static structure factor S(q) using the Percus-Yevick
    closure scheme.

    Parameters
    ----------
    phi : volume fraction of spheres
    qd : q-vector times the sphere diameter

    Returns
    -------
    S(q): static structure factor using the mean spherical approximation (MSA)

    References
    B. Hammouda, Probing nanoscale structures -- The SANS toolbox.
    https://www.ncnr.nist.gov/staff/hammouda/the_SANS_toolbox.pdf.  p.332.
    ----------

    """
    # For qd > qd_small, S(q) is calculated using the formulas given in
    # Trokhymchuk, I. Nezbeda, J. Jirsák, & D. Henderson.
    # For qd < qd_small, a Taylor expansion of S(q) about qd=0 is used.
    # Setting the value of qd_small=0.13 gives values of S(q) accurate to
    # 1e-9 below the cutoff, which is generally smaller than the round-off
    # errors incurred using the direct formula.
    qd_small = 0.13

    lambda1 = (1. + 2. * phi) ** 2. / (1. - phi) ** 4
    lambda2 = -(1. + 0.5 * phi) ** 2. / (1. - phi) ** 4
    qd2 = qd * qd
    qd3 = qd * qd2
    qd4 = qd2 * qd2
    qd6 = qd2 * qd4

    s, c = np.sin(qd), np.cos(qd)
    nc1 = lambda1 * (s - qd * c) / qd3
    nc1 += -6. * phi * lambda2 * (qd2 * c - 2. * qd * s - 2. * c + 2.) / qd4
    numerator = qd4 * c - 4. * qd3 * s - 12. * qd2 * c + 24. * qd * s + 24. * c - 24.
    nc1 += -0.5 * phi * lambda1 * numerator / qd6
    nc1 *= -24. * phi

    nc2 = -2. * (phi * (18. * lambda2 * phi + lambda1 * (4. + phi)))
    nc2 += phi * (4. * lambda2 * phi + lambda1 * (0.8 + 0.25 * phi)) * qd2
    nc2 += -phi * (0.15 * lambda2 * phi + lambda1 * (1. / 35. + 0.01 * phi)) * qd4

    return np.where(qd > qd_small, 1. / (1. - nc1), 1. / (1. - nc2))
